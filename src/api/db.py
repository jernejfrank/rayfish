"""Local database using sqlight to save simulation results."""

import platform
import sqlite3
import uuid
from pathlib import Path

import pandas as pd
from beartype.typing import Any, Dict, Optional, Tuple

# Public API
__all__ = ("SQLiteClient",)


class SQLiteClient:
    """Saves and retrieves simulation results from a local sqlite db."""

    @staticmethod
    def get_db_dir() -> Path:
        """Determines root directory and database subdirectory."""
        system = platform.system()
        if system == "Windows":
            data_dir = Path.home() / "Rayfish"
        else:
            data_dir = Path.home() / ".rayfish"
        return data_dir

    def __init__(self, custom_dir: Optional[str] = None):
        if custom_dir is not None:
            self.db_file = Path(custom_dir) / "db.sqlite"
        else:
            db_dir = SQLiteClient.get_db_dir()
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_file = db_dir / "db.sqlite"

    def init_main(self) -> None:
        """Initializes main table to keep track of simulations."""
        with self as conn:
            cursor = conn.cursor()
            cursor.execute("""
                            CREATE TABLE IF NOT EXISTS simulations (
                            id TEXT PRIMARY KEY,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            light_type TEXT,
                            estimation_parameter TEXT,
                            simulation_dimension INTEGER,
                            number_of_point_sources INTEGER,
                            wave_vector FLOAT,
                            rayleigh_length FLOAT,
                            number_of_point_source_distributions INTEGER,
                            maximal_length FLOAT,
                            minimal_spacing FLOAT
                            );
                           """)

    def __enter__(self):
        """Entry point for context manager."""
        self.connection = sqlite3.connect(self.db_file)
        return self.connection

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        """Cleanup."""
        if self.connection:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
            self.connection.close()

    def get_simulation_id(
        self, simulation_config: Dict[str, Any], input_parameters: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """Check if simulation already has table, otherwise create id for it."""
        all_param_keys = [f"{key} = ?" for key in simulation_config] + [
            f"{key} = ?" for key in input_parameters
        ]
        where_clause = " AND ".join(all_param_keys)
        all_param_values = list(simulation_config.values()) + list(
            input_parameters.values()
        )
        query = f"SELECT id FROM simulations WHERE {where_clause}"  # noqa: S608

        try:
            with self as conn:
                cursor = conn.cursor()
                cursor.execute(query, all_param_values)
                sim_id = cursor.fetchone()
        except sqlite3.OperationalError:
            sim_id = None

        if sim_id:
            return sim_id[0], True

        sim_id = str(uuid.uuid4())  # new simulation gets unique ID
        return sim_id, False

    def save_simulation_parameters(
        self,
        sim_id: str,
        simulation_config: Dict[str, Any],
        input_parameters: Dict[str, Any],
    ) -> None:
        """Saves simulation parameters in main database table."""
        joint_dict = simulation_config | input_parameters
        joint_dict["id"] = sim_id
        query = """INSERT INTO simulations (
                    id,
                    light_type,
                    estimation_parameter,
                    simulation_dimension,
                    number_of_point_sources,
                    wave_vector,
                    rayleigh_length,
                    number_of_point_source_distributions,
                    maximal_length,
                    minimal_spacing
                    ) VALUES (
                    :id,
                    :light_type,
                    :estimation_parameter,
                    :simulation_dimension,
                    :number_of_point_sources,
                    :wave_vector,
                    :rayleigh_length,
                    :number_of_point_source_distributions,
                    :maximal_length,
                    :minimal_spacing
                    )"""

        with self as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, joint_dict)
                conn.commit()

            # TODO: need to add logger to app
            except sqlite3.IntegrityError:
                pass
                # print(f"Error: {e}. "
                # "Simulation with id '{joint_dict['id']}' already exists.")

    def save_simulation_results(self, table_name: str, df: pd.DataFrame) -> None:
        """Saves dataframe to database."""
        with self as conn:
            df.to_sql(table_name, con=conn, if_exists="append")

    def load(self, table_name: str) -> pd.DataFrame:
        """Load dataframe from database."""
        with self as conn:
            df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)  # noqa: S608
        return df
