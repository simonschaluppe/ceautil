import os
from pathlib import Path
import pandas as pd
import logging as lg


def get_scenario_subdirs(simulation_dir_path):
    """
    Get the list of scenario subdirectories in the specified simulation directory.

    Parameters:
    simulation_dir_path (str or Path): The path to the simulation directory.

    Returns:
    list: A list of Path objects representing the scenario subdirectories.
    """
    scenario_paths = []
    simdir = Path(simulation_dir_path)
    for scenario_dir in simdir.iterdir():
        if scenario_dir.is_file():
            continue
        lg.info(f"Scenario found: \033[1m\033[94m{scenario_dir.name}\033[0m")
        scenario_paths.append(scenario_dir)
    if len(scenario_paths) == 0:
        lg.warning(f"No scenarios found in {Path(simdir)}.")
    return scenario_paths


def get_hull_df_for_simulation(simulation_folder_path):
    """
    Generate a DataFrame containing hull area data for the **first** scenario subfolder
    within the specified simulation folder.

    Parameters:
    simulation_folder_path (str or Path): The path to the simulation folder.

    Returns:
    pd.DataFrame: A DataFrame containing the hull areas for each building in the first scenario.
    """
    # Initialize an empty list to collect the data
    scenario_subdirs = get_scenario_subdirs(simulation_folder_path)
    geometry_path = (
        Path(simulation_folder_path)
        / scenario_subdirs[0]
        / "outputs"
        / "data"
        / "solar-radiation"
    )
    all_data = []
    # Loop through each file in the directory
    for filepath in geometry_path.iterdir():
        if not filepath.is_file():
            continue
        if not filepath.name.endswith("_geometry.csv"):
            continue

        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)
        # Group by 'orientation' and 'TYPE', and sum the 'AREA_m2' column
        grouped_df = df.groupby(["orientation", "TYPE"])["AREA_m2"].sum().reset_index()
        # Pivot the grouped DataFrame to have orientations and types as columns and AREA_m2 as values
        pivot_df = grouped_df.pivot_table(
            index=None, columns=["orientation", "TYPE"], values="AREA_m2", fill_value=0
        )
        # Flatten the columns
        pivot_df.columns = [
            f"{col[0].lower()}_{col[1].lower()}" for col in pivot_df.columns
        ]
        # Add the building ID as a column
        pivot_df["ID"] = filepath.name.split("_")[0]
        # Append the pivoted DataFrame to the list
        all_data.append(pivot_df)
    # Concatenate all the dataframes in the list into a single DataFrame
    final_df = pd.concat(all_data, ignore_index=True).set_index("ID")
    return final_df


def aggregate_orientations(cea_hull_df, inplace=False) -> pd.DataFrame:
    """
    Aggregate orientations (walls, windows, and roofs) from the provided DataFrame.

    Parameters:
    cea_hull_df (pd.DataFrame): The input DataFrame containing orientation-specific columns.
    inplace (bool): If True, modifies the input DataFrame in place. Otherwise, returns a new DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with aggregated wall, window, and roof areas, along with the total hull above ground 'hull_ag'.
    """
    cols = cea_hull_df.columns
    walls = [col for col in cols if "_walls" in col]
    windows = [col for col in cols if "_windows" in col]
    roofs = [col for col in cols if "_roofs" in col]

    df = pd.DataFrame()

    df["walls"] = cea_hull_df[walls].sum(axis=1)
    df["windows"] = cea_hull_df[windows].sum(axis=1)
    df["roofs"] = cea_hull_df[windows].sum(axis=1)
    df["hull_ag"] = df.sum(axis=1)
    if inplace:
        for col in df.columns:
            cea_hull_df[col] = df[col]
        return cea_hull_df
    return df


def get_annual_demands_for_simulation(simulation_folder_path):
    """
    Get the annual demands for all buildings for all scenarios in the specified simulation folder.

    Parameters:
    simulation_folder_path (str or Path): The path to the simulation folder.

    Returns:
    pd.DataFrame: A DataFrame containing the combined annual demand data for all buildings,
    the scenario name is in column "scenario".
    """
    dfs = []
    for scenario_dir in get_scenario_subdirs(simulation_folder_path):
        lg.info(f"{scenario_dir.name} found!")
        dfs.append(get_annual_demand_for_scenario(scenario_dir))
    return pd.concat(dfs, ignore_index=True)


def get_annual_demand_for_scenario(scenario_dir):
    """
    Get the annual demand for a single scenario directory.

    Parameters:
    scenario_dir (Path): The path to the scenario directory.

    Returns:
    pd.DataFrame: A DataFrame containing the demand data for the specified scenario.
    """
    total_demand_file = (
        scenario_dir / "outputs" / "data" / "demand" / "Total_demand.csv"
    )
    lg.info(f"{total_demand_file.name} found in {scenario_dir.name}")
    df = pd.read_csv(total_demand_file)
    df.insert(
        1,
        "scenario",
        scenario_dir.name,
    )
    return df


def as_area_specific(source_df, value_cols: list = None, area_col="Af_m2"):
    """
    Convert absolute values in the source DataFrame to area-specific values.
    Also converts MWh to kWh.

    Parameters:
    source_df (pd.DataFrame): The input DataFrame with absolute values.
    value_cols (list): A list of column names to be converted. If None, all numeric columns are selected.
    area_col (str): The column representing the area to normalize by.

    Returns:
    pd.DataFrame: A copy of the source_df with area-specific values for the specified columns.
    """
    # If value_cols is None, select all numeric columns
    if value_cols is None:
        value_cols = source_df.select_dtypes(include="number").columns
    else:
        # Only keep numeric columns from the provided value_cols
        value_cols = [
            col for col in value_cols if pd.api.types.is_numeric_dtype(source_df[col])
        ]

    data = {}
    for col in value_cols:
        if "m2" in col:
            continue
        parts = col.split("_")
        name = "_".join(parts[:-1])
        unit = parts[-1]
        factor = 1
        unit = unit + "_" + area_col
        if "kW" in unit:
            unit = unit.replace("kW", "W")
            factor = 1000
        if "MWh" in unit:
            unit = unit.replace("MWh", "kWh")
            factor = 1000

        data[name + "_" + unit] = source_df[col] * factor / source_df[area_col]
    non_value_columns = source_df.columns.difference(value_cols)
    for nc in non_value_columns:
        data[nc] = source_df[nc]
    return pd.DataFrame(data)


def get_compactness(
    hull_df, area_df, hull_cols=["walls", "windows", "hull_ag"], ref_col="GFA_m2"
):
    """
    Calculate the compactness of buildings by dividing hull columns by a given reference area.

    Parameters:
    hull_df (pd.DataFrame): The input DataFrame containing hull data.
    area_df (pd.DataFrame): The DataFrame containing area information (GFA).
    hull_cols (list): The columns from hull_df to use for compactness calculation. (default: ["walls", "windows", "hull_ag"])
    ref_col (str): The column in area_df representing the reference area (default: 'GFA_m2').

    Returns:
    pd.DataFrame: A DataFrame with compactness values for each of (wall, window, hull) columns to the reference area.
    """
    GFAs = area_df[ref_col]
    GFAs = GFAs.groupby(GFAs.index).agg("mean")
    GFAsfirst = area_df[ref_col].loc[~area_df.index.duplicated(keep="first")]
    if not GFAsfirst.equals(GFAs):
        lg.warning(
            "the first buidling areas dont match the aggregates for the building!"
        )
    if len(GFAs) != len(hull_df):
        lg.warning(
            f"The number of rows in hull_df ({len(hull_df)}) and GFAs ({len(GFAs)}) don't match!"
        )
    d = {}
    for col in hull_cols:
        d[col + "_to_" + ref_col] = hull_df[col] / GFAs
    return pd.DataFrame(d)


def combine_results(simulation_dir):
    """
    Combine results from multiple scenario outputs into a single DataFrame.

    This function aggregates various building data from a specified simulation directory
    and combines them into a single DataFrame. It includes oriented and aggregated hull
    data, compactness calculations and area-specific demand data, for each scenario.

    Parameters:
        simulation_dir (str or Path): The directory containing the simulation outputs.

    Returns:
        pd.DataFrame: A DataFrame combining hull geometry, orientations, area-specific demand data, and compactness for all building scenarios.
    """

    df_hull_oriented = get_hull_df_for_simulation(simulation_dir)
    df_hull_aggregated = aggregate_orientations(df_hull_oriented)
    demands = get_annual_demands_for_simulation(simulation_dir).set_index("Name")
    demands_m2 = as_area_specific(demands)
    df_compact = get_compactness(df_hull_aggregated, demands)

    # fill all building scenarios with static data
    df_hull_oriented_exp = df_hull_oriented.reindex(demands_m2.index)
    df_hull_aggregated_exp = df_hull_aggregated.reindex(demands_m2.index)
    df_compact_expanded_exp = df_compact.reindex(demands_m2.index)
    # Concatenate the expanded df_compact and demands_m2 along the columns (axis=1)
    df = pd.DataFrame()
    df = pd.concat(
        [
            df_hull_oriented_exp,
            df_hull_aggregated_exp,
            df_compact_expanded_exp,
            demands[["Af_m2", "Aroof_m2", "GFA_m2", "Aocc_m2"]],
            demands_m2,
        ],
        axis=1,
    )
    return df


def add_compactness_category_col(
    df, bins, labels, col="hull_ag_to_GFA_m2", name="compact_category"
):
    if len(bins) != len(labels) + 1:
        raise ValueError(
            f"There must be one more bins ({len(bins)}) than ({len(labels)}))!"
        )
    df[name] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df


if __name__ == "__main__":
    lg.basicConfig(level=lg.INFO)
    simdir = r"C:\Users\Simon Schneider\Nextcloud\EE\1_Forschung\2_Laufend\2023 MA25 Alliiertenviertel\FHTW Ergebnisse\20241022 CEA run part"
    subdir = Path("outputs/data/solar-radiation")
    print(get_scenario_subdirs(simdir))
    hulls = get_hull_df_for_simulation(simdir)
    hulls.head()
    demands = get_annual_demands_for_simulation(simdir)
    demands.columns
