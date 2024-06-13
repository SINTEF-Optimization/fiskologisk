import sys
import json

def char_diagram(file_path: str) -> None:

    input_file = open(file_path, "r")
    data = json.load(input_file)

    row_module = {}
    row_tank = {}
    module_tank_idx = {}

    table_rows = []

    row = 0
    for m in data["modules"]:
        m_idx = m["module_index"]
        is_first = True
        row_module[m["module_index"]] = row
        mod_t_idx = 0
        for t in m["tank_indices"]:
            row_str = "{: >2}".format(str(m["module_index"])) if is_first else "  "
            row_str += "{: >3}".format(str(mod_t_idx)) + " "
            table_rows.append(row_str)
            row_tank[t] = row
            module_tank_idx[t] = mod_t_idx
            row += 1
            mod_t_idx += 1
            is_first = False

    years_str = "      "
    months_str = "      "
    dep_m_str = "Md Tk "

    pl_hor = data["planning_horizon"]
    year = pl_hor["first_year"]
    first_period = pl_hor["first_period"]
    last_period = pl_hor["last_period"]
    last_ordinary_period = pl_hor["last_ordinary_horizon_period"]

    period_col = {}
    column = 6
    months_chars = ["M", "A", "M", "J", "J", "A", "S", "O", "N", "D", "J", "F"]

    for period in range(first_period, last_period + 1):
        month_in_year = period % 12
        if month_in_year == 0:
            extended_start = period > last_ordinary_period and period - 12 <= last_ordinary_period
            spaces = " " * (3 if extended_start else 1)
            if extended_start:
                extend_barrier_column = column + 1

            years_str += spaces
            if period > last_ordinary_period:
                years_str += "Ext-" + "{: <8}".format(str(year))
            else:
                years_str += "{: <12}".format(str(year))

            months_str += spaces
            dep_m_str += spaces
            for idx in range(len(table_rows)):
                table_rows[idx] += spaces

            column += len(spaces)
            year += 1

        months_str += months_chars[month_in_year]
        dep_m_str += "-"
        for idx in range(len(table_rows)):
            table_rows[idx] += "."

        period_col[period] = column
        column += 1

    for dep_p in pl_hor["deploy_periods"]:
        dep_m_str = replace(dep_m_str, "#", period_col[dep_p])

    for prod_cyc in data["production_cycles"]:
        for tank_cyc in prod_cyc["tank_cycles"]:
            t = tank_cyc["tank"]
            start_p = tank_cyc["start_period"]
            start_cause = tank_cyc["start_cause"]
            end_p = tank_cyc["end_period"]
            end_cause = tank_cyc["end_cause"]
            row = row_tank[t]
            if start_p <= last_ordinary_period and end_p > last_ordinary_period:
                table_rows[row] = replace(table_rows[row], ">", extend_barrier_column)

            for p in range(start_p, end_p + 1):
                table_rows[row] = replace(table_rows[row], "x", period_col[p])

            if start_cause == "pre_planning_deploy":
                table_rows[row] = replace(table_rows[row], "*", period_col[start_p] - 1)
            elif start_cause == "deploy":
                table_rows[row] = replace(table_rows[row], "D", period_col[start_p])
            elif start_cause == "transfer":
                transfer = tank_cyc["transfer"]
                from_t = transfer["from_tank"]
                tr_period_col = period_col[transfer["period"]]
                from_row = row_tank[from_t]
                table_rows[row] = replace(table_rows[row], str(module_tank_idx[from_t]), tr_period_col)
                table_rows[from_row] = replace(table_rows[from_row], "t", tr_period_col)

            if end_cause == "post_smolt":
                table_rows[row] = replace(table_rows[row], "S", period_col[end_p])
            elif end_cause == "harvest":
                table_rows[row] = replace(table_rows[row], "H", period_col[end_p])

    print(years_str)
    print(months_str)
    print(dep_m_str)
    for tab_row in table_rows:
        print(tab_row)

def replace(in_string: str, new_ch: str, position: int) -> str:
    return in_string[:position] + new_ch + in_string[position+1:]

if __name__ == "__main__":
    char_diagram(sys.argv[1])
