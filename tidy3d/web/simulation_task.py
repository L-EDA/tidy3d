"""Tidy3d webapi types."""
from __future__ import annotations

import os.path
import tempfile
from datetime import datetime
from typing import List, Optional, Callable, Tuple
import pydantic as pd
from pydantic import Extra, Field, parse_obj_as

from tidy3d import Simulation
from tidy3d.version import __version__

from .cache import FOLDER_CACHE
from .http_management import http
from .s3utils import download_file, upload_file, upload_string
from .types import Queryable, ResourceLifecycle, Submittable
from .types import Tidy3DResource

SIMULATION_JSON = "simulation.json"
SIMULATION_HDF5 = "output/monitor_data.hdf5"
RUNNING_INFO = "output/solver_progress.csv"
LOG_FILE = "output/tidy3d.log"
SIM_FILE_HDF5 = "simulation.hdf5"


class Folder(Tidy3DResource, Queryable, extra=Extra.allow):
    """Tidy3D Folder."""

    folder_id: str = Field(..., title="Folder id", description="folder id", alias="projectId")
    folder_name: str = Field(
        ..., title="Folder name", description="folder name", alias="projectName"
    )

    # pylint: disable=arguments-differ
    @classmethod
    def list(cls) -> []:
        """List all folders.

        Returns
        -------
        folders : [Folder]
            List of folders
        """
        resp = http.get("tidy3d/projects")
        return (
            parse_obj_as(
                List[Folder],
                resp,
            )
            if resp
            else None
        )

    # pylint: disable=arguments-differ
    @classmethod
    def get(cls, folder_name: str):
        """Get folder by name.

        Parameters
        ----------
        folder_name : str
            Name of the folder.

        Returns
        -------
        folder : Folder
        """
        resp = http.get(f"tidy3d/project?projectName={folder_name}")
        return Folder(**resp) if resp else None

    # pylint: disable=arguments-differ
    @classmethod
    def create(cls, folder_name: str):
        """Create a folder, return existing folder if there is one has the same name.

        Parameters
        ----------
        ----------
        folder_name : str
            Name of the folder.

        Returns
        -------
        folder : Folder
        """
        folder = Folder.get(folder_name)
        if folder:
            return folder
        resp = http.post("tidy3d/projects", {"projectName": folder_name})
        return Folder(**resp) if resp else None

    def delete(self):
        """Remove this folder."""

        http.delete(f"tidy3d/projects/{self.folder_id}")

    def list_tasks(self) -> List[Tidy3DResource]:
        """List all tasks in this folder.

        Returns
        -------
        tasks : List[:class:`.SimulationTask`]
            List of tasks in this folder
        """
        resp = http.get(f"tidy3d/projects/{self.folder_id}/tasks")
        return (
            parse_obj_as(
                List[SimulationTask],
                resp,
            )
            if resp
            else None
        )


class SimulationTask(ResourceLifecycle, Submittable, extra=Extra.allow):
    """Interface for managing the running of a :class:`.Simulation` task on server."""

    task_id: Optional[str] = Field(
        ...,
        title="task_id",
        description="Task ID number, set when the task is uploaded, leave as None.",
        alias="taskId",
    )
    status: Optional[str] = Field(title="status", description="Simulation task status.")

    real_flex_unit: float = Field(
        None, title="real flex units", description="Billed flex units.", alias="realCost"
    )

    created_at: Optional[datetime] = Field(
        title="created_at", description="Time at which this task was created.", alias="createdAt"
    )

    simulation: Optional[Simulation] = Field(
        title="simulation", description="A copy of the Simulation being run as this task."
    )

    folder_name: Optional[str] = Field(
        "default",
        title="Folder Name",
        description="Name of the folder associated with this task.",
        alias="projectName",
    )

    folder: Optional[Folder]

    callback_url: str = Field(
        None,
        title="Callback URL",
        description="Http PUT url to receive simulation finish event. "
        "The body content is a json file with fields "
        "``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.",
    )

    @pd.root_validator(pre=True)
    def _error_if_jax_sim(cls, values):
        """Raise error if user tries to submit simulation that's a JaxSimulation."""
        sim = values.get("simulation")
        if sim is None:
            return values
        if "JaxSimulation" in str(type(sim)):
            raise ValueError(
                "'JaxSimulation' not compatible with regular webapi functions. "
                "Either convert it to Simulation with 'jax_sim.to_simulation()[0]' or use "
                "the 'adjoint.run' function to run JaxSimulations."
            )
        return values

    # pylint: disable=arguments-differ
    @classmethod
    def create(
        cls, simulation: Simulation, task_name: str, folder_name="default", call_back_url=None
    ) -> SimulationTask:
        """Create a new task on the server.

        Parameters
        ----------
        simulation: :class".Simulation"
            The :class:`.Simulation` will be uploaded to server in the submitting phase.
            If Simulation is too large to fit into memory, pass None to this parameter
            and use :meth:`.simulationTask.upload_file` instead.
        task_name: str
            The name of the task.
        folder_name: str,
            The name of the folder to store the task. Default is "default".
        call_back_url: str
            Http PUT url to receive simulation finish event. The body content is a json file with
            fields ``{'id', 'status', 'name', 'workUnit', 'solverVersion'}``.

        Returns
        -------
        :class:`SimulationTask`
            :class:`SimulationTask` object containing info about status, size,
             credits of task and others.
        """
        folder = FOLDER_CACHE.get(folder_name)
        if not folder:
            folder = Folder.get(folder_name)
        if not folder:
            folder = Folder.create(folder_name)
        FOLDER_CACHE[folder_name] = folder

        resp = http.post(
            f"tidy3d/projects/{folder.folder_id}/tasks",
            {"taskName": task_name, "call_back_url": call_back_url},
        )

        return SimulationTask(**resp, simulation=simulation, folder=folder)

    # pylint: disable=arguments-differ
    @classmethod
    def get(cls, task_id: str) -> SimulationTask:
        """Get task from the server by id.

        Parameters
        ----------
        task_id: str
            Unique identifier of task on server.

        Returns
        -------
        :class:`.SimulationTask`
            :class:`.SimulationTask` object containing info about status,
             size, credits of task and others.
        """
        resp = http.get(f"tidy3d/tasks/{task_id}/detail")
        return SimulationTask(**resp) if resp else None

    def delete(self):
        """Delete current task from server."""
        if not self.task_id:
            raise ValueError("Task id not found.")
        http.delete(f"tidy3d/tasks/{self.task_id}")

    def get_simulation(self) -> Optional[Simulation]:
        """Download simulation from server.

        Returns
        -------
        :class:`.Simulation`
            :class:`.Simulation` object containing info about status, size,
             credits of task and others.
        """
        if self.simulation:
            return self.simulation

        with tempfile.NamedTemporaryFile(suffix=".json") as temp:
            self.get_simulation_json(temp.name)
            if os.path.exists(temp.name):
                self.simulation = Simulation.from_file(temp.name)
                return self.simulation
        return None

    def get_simulation_json(
        self, to_file: str, verbose: bool = True, progress_callback: Callable[[float], None] = None
    ) -> None:
        """Get json file for a :class:`.Simulation` from server.

        Parameters
        ----------
        to_file: str
            save file to path.
        """
        assert self.task_id
        download_file(
            self.task_id,
            SIMULATION_JSON,
            to_file=to_file,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def upload_simulation(
        self, verbose: bool = True, progress_callback: Callable[[float], None] = None
    ) -> None:
        """Upload :class:`.Simulation` object to Server.

        Parameters
        ----------
        verbose: bool = True
            Whether to display progressbars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called when uploading file.
        """
        assert self.task_id
        assert self.simulation
        upload_string(
            self.task_id,
            self.simulation.json(),
            SIMULATION_JSON,
            verbose=verbose,
            progress_callback=progress_callback,
        )
        if len(self.simulation.custom_datasets) > 0:
            # Also upload hdf5 containing all data.
            # The temp file will be re-opened in `to_hdf5` which can cause an error on some systems
            # so we explicitly close it first.
            data_file = tempfile.NamedTemporaryFile()  # pylint:disable=consider-using-with
            data_file.close()
            self.simulation.to_hdf5(data_file.name)
            upload_file(
                self.task_id,
                data_file.name,
                SIM_FILE_HDF5,
                verbose=verbose,
                progress_callback=progress_callback,
            )

    def upload_file(
        self,
        local_file: str,
        remote_filename: str,
        verbose: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> None:
        """
        Upload file to platform. Using this method when the json file is too large to parse
         as :class".simulation".
        Parameters
        ----------
        local_file: str
            local file path.
        remote_filename: str
            file name on the server
        verbose: bool = True
            Whether to display progressbars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called when uploading file.
        """
        assert self.task_id

        upload_file(
            self.task_id,
            local_file,
            remote_filename,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def submit(
        self,
        solver_version: str = None,
        worker_group: str = None,
    ):
        """Kick off this task. If this task instance contain a :class:`.Simulation`,
         it will be uploaded to server first,
        then kick off the task. Otherwise, this method makes assumption that
         the Simulation has been uploaded by the
        upload_file function, so the task will be kicked off directly.

        Parameters
        ----------
        solver_version: str = None
            target solver version.
        worker_group: str = None
            worker group
        """
        if self.simulation:
            upload_string(self.task_id, self.simulation.json(), SIMULATION_JSON, verbose=False)

        if solver_version:
            protocol_version = None
        else:
            protocol_version = __version__

        http.post(
            f"tidy3d/tasks/{self.task_id}/submit",
            {
                "solverVersion": solver_version,
                "workerGroup": worker_group,
                "protocolVersion": protocol_version,
            },
        )

    def estimate_cost(self, solver_version=None) -> float:
        """Compute the maximum flex unit charge for a given task, assuming the simulation runs for
        the full ``run_time``. If early shut-off is triggered, the cost is adjusted proportionately.

        Parameters
        ----------
        solver_version: str
            target solver version.

        Returns
        -------
        flex_unit_cost: float
            estimated cost in flex units
        """

        if solver_version:
            protocol_version = None
        else:
            protocol_version = __version__

        assert self.task_id
        resp = http.post(
            f"tidy3d/tasks/{self.task_id}/metadata",
            {
                "solverVersion": solver_version,
                "protocolVersion": protocol_version,
            },
        )
        return resp

    def get_simulation_hdf5(
        self, to_file: str, verbose: bool = True, progress_callback: Callable[[float], None] = None
    ) -> None:
        """Get hdf5 file from Server.

        Parameters
        ----------
        to_file: str
            save file to path.
        verbose: bool = True
            Whether to display progressbars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called when uploading file.
        """
        assert self.task_id
        download_file(
            self.task_id,
            SIMULATION_HDF5,
            to_file=to_file,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def get_running_info(self) -> Tuple[float, float]:
        """Gets the % done and field_decay for a running task.

        Returns
        -------
        perc_done : float
            Percentage of run done (in terms of max number of time steps).
            Is ``None`` if run info not available.
        field_decay : float
            Average field intensity normlized to max value (1.0).
            Is ``None`` if run info not available.
        """
        assert self.task_id

        resp = http.get(f"tidy3d/tasks/{self.task_id}/progress")
        perc_done = resp.get("perc_done")
        field_decay = resp.get("field_decay")
        return perc_done, field_decay

    def get_log(
        self, to_file: str, verbose: bool = True, progress_callback: Callable[[float], None] = None
    ) -> None:
        """Get log file from Server.

        Parameters
        ----------
        to_file: str
            save file to path.
        verbose: bool = True
            Whether to display progressbars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called when uploading file.
        """

        assert self.task_id
        download_file(
            self.task_id,
            LOG_FILE,
            to_file=to_file,
            verbose=verbose,
            progress_callback=progress_callback,
        )