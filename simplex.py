import argparse
import numpy as np
import os
import sys
import io
import traceback

class SimplexArgs:
    def __init__(self):
        self.filename = ""  
        self.decimals = 3
        self.digits = 7  # Updated default to match argparse
        self.policy = "largest"

class SimplexConfig:
    def __init__(self):
        self.epsilon = 1e-12  # Numeric precision for comparisons
        self.number_format = "{:>7.3f}"  # Temporary default; will be updated later

# Global instances for arguments and config
SARGS = SimplexArgs()
CONFIG = SimplexConfig()

def parse_cli_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Método Simplex para Programação Linear",
        epilog="Desenvolvido para a disciplina de Pesquisa Operacional",
    )

    parser.add_argument(
        "filename",
        type=str,
        nargs="?",
        default=None,
        help="Arquivo de entrada no formato LP",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Casas decimais na saída",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=7,
        help="Largura total dos números",
    )
    parser.add_argument(
        "--policy",
        choices=["largest", "bland", "smallest"],
        default="largest",
        help="Regra de escolha do pivô: largest (maior), bland (primeiro), smallest (menor)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Executa suite de testes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra detalhes durante os testes",
    )

    args = parser.parse_args()
    if not args.test and args.filename is None:
        parser.error("Arquivo de entrada é obrigatório quando não está em modo de teste")

    return args

def print_matrix(mat):
    """Print a matrix with proper alignment and borders."""
    formatted = [[CONFIG.number_format.format(val) for val in row] for row in mat]
    col_widths = [
        max(len(formatted[r][c]) for r in range(len(mat))) for c in range(len(mat[0]))
    ]
    total_width = sum(col_widths[:-1]) + 3 * (len(col_widths) - 1)
    rhs_width = col_widths[-1]
    split_pos = len(mat) - 1  # Position after slack variables

    print("-" * total_width + "-" * 5 + "-" * rhs_width)
    for i, row in enumerate(formatted):
        if i == 1:
            print("-" * total_width + "-" * 5 + "-" * rhs_width)
        print("|", end=" ")
        for j, (value, width) in enumerate(zip(row, col_widths)):
            if j == split_pos:
                print("||", end=" ")
            elif j == len(row) - 1:
                print("|", end=" ")
            print(value.rjust(width), end=" ")
        print("|")
    print("-" * total_width + "-" * 5 + "-" * rhs_width)

def parse_objective_function(f, num_vars, var_types):
    """
    Reads and processes the objective function from the file.
    var_types: 1 = x >= 0; -1 = x <= 0; 0 = free
    """
    line = f.readline().split()
    obj_sign = -1 if line[0] == "min" else 1

    coefs = []
    for i in range(num_vars):
        coef = int(line[i + 1]) * obj_sign

        # For negative variables or free variables, adjust sign
        if var_types[i] == -1:
            coef *= -1
        elif var_types[i] == 0:
            # For free variables, we effectively split them into x+ and x-
            coefs.append(-int(line[i + 1]) * obj_sign)

        coefs.append(coef)

    return coefs

def parse_constraints(f, num_cons, num_vars, var_types):
    """
    Reads and processes the constraints from the file.
    var_types: 1 = x >= 0; -1 = x <= 0; 0 = free
    """
    mat = []
    rhs = []
    eq_flags = []

    for _ in range(num_cons):
        parts = f.readline().split()
        sign = 1 if parts[-2] != ">=" else -1
        eq_flags.append(1 if parts[-2] == "==" else 0)

        row = []
        for j in range(len(parts) - 2):
            val = int(parts[j]) * sign
            if var_types[j] == -1:
                val *= -1
            elif var_types[j] == 0:
                # For free variable split
                row.append(-int(parts[j]) * sign)
            row.append(val)

        mat.append(row)
        rhs.append(int(parts[-1]) * sign)

    # Adjust number of variables for free variables + 1 for RHS
    num_vars += var_types.count(0) + 1
    return mat, rhs, eq_flags, num_vars

def convert_problem_to_standard_form(num_rows, num_cols, mat, obj_coefs, rhs, eq_flags):
    """
    Converts the problem to standard form (adds slack variables and RHS column).
    eq_flags[i] = 1 if equality constraint, 0 otherwise.
    """
    for i in range(len(eq_flags)):
        if not eq_flags[i]:
            obj_coefs.append(0)
            num_cols += 1
            for r in range(num_rows):
                mat[r].append(1 if i == r else 0)

    for r in range(num_rows):
        mat[r].append(rhs[r])

    return num_rows, num_cols, mat, obj_coefs, rhs

def make_matrix_full_rank(A):
    """
    Removes linearly dependent rows from matrix.
    Provided by Professor Cristiano Arbex.
    """
    if np.linalg.matrix_rank(A) == A.shape[0]:
        return A, []

    row_idx = 1
    rows_eliminated = []
    counter = 0
    while True:
        counter += 1
        B = A[: row_idx + 1, :]
        C = np.linalg.qr(B.T)[1]
        C[np.isclose(C, 0)] = 0
        if row_idx >= C.shape[0]:
            break
        if not np.any(C[row_idx, :]):
            rows_eliminated.append(counter - 1)
            A = np.delete(A, row_idx, axis=0)
        else:
            row_idx += 1
        if row_idx >= A.shape[0]:
            break

    return A, rows_eliminated

def parse_input():
    """Reads all input data from the specified file."""
    with open(SARGS.filename, "r") as f:
        num_vars = int(f.readline().strip())
        num_cons = int(f.readline().strip())
        var_types = list(map(int, f.readline().split()))

        objective = parse_objective_function(f, num_vars, var_types)
        mat, rhs, eq_flags, num_vars = parse_constraints(f, num_cons, num_vars, var_types)

        # Check if we can remove dependent rows
        eliminated_rows = []
        mat_array = np.array(mat)
        original_rank = np.linalg.matrix_rank(mat_array)
        if original_rank < mat_array.shape[0]:
            new_mat_array, eliminated_rows = make_matrix_full_rank(mat_array)
            # Only apply elimination if rank improved
            if np.linalg.matrix_rank(new_mat_array) > original_rank:
                mat_array = new_mat_array
            else:
                eliminated_rows = []

        if eliminated_rows:
            mat = mat_array.tolist()
            rhs = [rhs[i] for i in range(len(rhs)) if i not in eliminated_rows]
            eq_flags = [eq_flags[i] for i in range(len(eq_flags)) if i not in eliminated_rows]
            num_cons = len(mat)

        # Convert to standard form
        num_cons, num_vars, mat, objective, rhs = convert_problem_to_standard_form(
            num_cons, num_vars, mat, objective, rhs, eq_flags
        )

    print("Matriz de Restrições:")
    print_matrix(mat)
    print("\nVetor RHS:")
    print(np.array(rhs))
    print("\nFunção Objetivo:")
    print(np.array(objective))
    print()

    return num_cons, num_vars, objective, mat, var_types

def initialize_tableau(rows, cols, obj_coefs, mat):
    """Creates the initial simplex tableau."""
    tableau = np.zeros((rows + 1, cols + rows), dtype=float)

    # Identity matrix on the left
    for i in range(rows):
        tableau[i + 1, i] = 1

    # Objective row (negative for maximization in tableau)
    for c in range(cols - 1):
        tableau[0, c + rows] = -obj_coefs[c]

    # Constraints
    for r in range(rows):
        for c in range(cols):
            tableau[r + 1, c + rows] = mat[r][c]

    cols += rows
    rows += 1

    print("Tableau Inicial:")
    print_matrix(tableau)
    print()

    return tableau, rows, cols

def identify_basis_vars(tableau, rows, cols):
    """Identifies basic and non-basic variables in the tableau."""
    non_basic = set(range(1, rows))  # row indices for potential basic variables
    basic = []

    for col in range(rows - 1, cols - 1):
        count_ones = 0
        row_idx = -1
        for r in range(1, rows):
            val = tableau[r, col]
            # Must be exactly 0 or 1 (with tolerance) for potential basis
            if val < -CONFIG.epsilon or val > 1 + CONFIG.epsilon:
                count_ones = 0
                break
            if abs(val - 1) < CONFIG.epsilon:
                count_ones += 1
                row_idx = r

        if count_ones == 1 and row_idx in non_basic:
            non_basic.remove(row_idx)
            basic.append(col)
            # Make the objective row consistent with this basis
            tableau[0] -= tableau[0, col] * tableau[row_idx]

    return tableau, non_basic, basic

def get_pivot_column(tableau, rows, cols):
    """
    Selects the pivot column according to the policy defined in SARGS.policy.
    
    Policies:
        - bland: first negative coefficient
        - smallest: most negative coefficient
        - largest: negative coefficient with the largest magnitude
    """
    best_val = 0
    pivot_col = -1

    for col in range(rows - 1, cols - 1):
        val = tableau[0, col]
        if val < -CONFIG.epsilon:
            if SARGS.policy == "bland":
                return col
            elif (SARGS.policy == "smallest" and (best_val == 0 or val < best_val)) or \
                 (SARGS.policy == "largest"  and (best_val == 0 or abs(val) > abs(best_val))):
                best_val = val
                pivot_col = col

    return pivot_col

def pivot(tableau, row_p, col_p):
    """
    Performs the pivot operation on the tableau:
      1. Normalize the pivot row
      2. Zero out the pivot column in other rows
    """
    tableau[row_p] /= tableau[row_p, col_p]
    pivot_val = tableau[row_p, col_p]
    for r in range(len(tableau)):
        if r != row_p and abs(tableau[r, col_p]) > CONFIG.epsilon:
            factor = tableau[r, col_p]
            tableau[r] -= factor * tableau[row_p]

def dual_phase(tableau, rows, cols):
    """
    Executes the dual phase of the simplex method.
      1. Find row with negative RHS
      2. Choose pivot column (most negative coefficient in that row)
      3. Pivot
      4. Repeat until all RHS >= 0
    Returns updated tableau and status ("inviavel" or None).
    """
    print("Fase Dual:")
    while True:
        pivot_row = -1
        pivot_col = -1
        min_rhs = 0
        # Find row with negative RHS in the last column
        for r in range(1, rows):
            if tableau[r, -1] < -CONFIG.epsilon and (min_rhs == 0 or tableau[r, -1] < min_rhs):
                min_rhs = tableau[r, -1]
                pivot_row = r

        if pivot_row < 0:
            print("Fase Dual Concluída\n")
            break

        # Find pivot column in the chosen row (most negative coefficient)
        min_coef = 0
        for c in range(rows - 1, cols - 1):
            val = tableau[pivot_row, c]
            if val < -CONFIG.epsilon and (min_coef == 0 or val < min_coef):
                min_coef = val
                pivot_col = c

        if pivot_col < 0:
            print()
            return tableau, "inviavel"

        print(f"Pivoteamento: linha {pivot_row}, coluna {pivot_col}\n")
        pivot(tableau, pivot_row, pivot_col)
        print_matrix(tableau)
        print()

    return tableau, None

def primal_phase(tableau, rows, cols):
    """
    Executes the primal phase of the simplex method.
      1. Choose pivot column (first or most negative coefficient in objective)
      2. Choose pivot row (minimum ratio test)
      3. Pivot
      4. Repeat until no negative coefficients remain
    Returns updated tableau and status ("otima" or "ilimitada").
    """
    print("Fase Primal:")
    while True:
        min_obj_val = 0
        pivot_col = -1
        # Find the most negative coefficient in the objective row
        for c in range(rows - 1, cols - 1):
            val = tableau[0, c]
            if val < -CONFIG.epsilon and (min_obj_val == 0 or val < min_obj_val):
                min_obj_val = val
                pivot_col = c

        if pivot_col < 0:
            print("Fase Primal Concluída\n")
            break

        # Minimum ratio test
        min_ratio = float("inf")
        pivot_row = -1
        for r in range(1, rows):
            if tableau[r, pivot_col] > CONFIG.epsilon:
                ratio = tableau[r, -1] / tableau[r, pivot_col]
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row = r

        if pivot_row < 0:
            print()
            return tableau, "ilimitada"

        print(f"Pivoteamento: linha {pivot_row}, coluna {pivot_col}\n")
        pivot(tableau, pivot_row, pivot_col)
        print_matrix(tableau)
        print()

    return tableau, "otima"

def get_primal_solution(tableau, rows, cols, var_types):
    """
    Extracts the final solution from the final tableau.
    Reconstructs solution for free variables by subtracting their negative counterpart.
    """
    tableau, non_basic, basic = identify_basis_vars(tableau, rows, cols)
    sol_map = {}
    for col in basic:
        row_idx = -1
        for r in range(1, rows):
            # Identify the row where the column is 1
            if abs(tableau[r, col] - 1) < CONFIG.epsilon:
                row_idx = r
                break
        if row_idx >= 0:
            sol_map[col] = tableau[row_idx, -1]

    # Build final solution considering variable types
    final_sol = [0] * len(var_types)
    j = rows - 1
    for i in range(len(var_types)):
        if var_types[i] == 0:
            # Free variable = x+ - x-
            val_pos = sol_map.get(j, 0)
            val_neg = sol_map.get(j + 1, 0)
            final_sol[i] = val_pos - val_neg
            j += 1
        else:
            final_sol[i] = sol_map.get(j, 0)
        j += 1

    return final_sol

def get_shadow_prices(tableau, rows):
    """
    Returns the dual solution (shadow prices) from the objective row
    corresponding to constraints in the primal problem.
    """
    return tableau[0, : rows - 1]

def build_aux_problem(tableau, rows, cols, non_basic_vars):
    """
    Builds the auxiliary problem:
      - Zero original objective
      - Add artificial variables for non-basic constraints
      - Minimizes the sum of artificial variables
    """
    rhs_values = tableau[:, -1].copy()
    aux_tableau = tableau[:, :-1].copy().tolist()

    # Zero the objective
    for c in range(rows - 1, cols - 1):
        aux_tableau[0][c] = 0

    # Add artificial variables
    for nbv in non_basic_vars:
        aux_tableau[0].append(1)
        for r in range(1, rows):
            aux_tableau[r].append(1 if nbv == r else 0)

    # Add RHS column
    for r in range(rows):
        aux_tableau[r].append(rhs_values[r])

    aux_tableau = np.array(aux_tableau)

    # Adjust objective by subtracting rows for each artificial var
    for nbv in non_basic_vars:
        aux_tableau[0] -= aux_tableau[nbv]

    return aux_tableau, rows, cols + len(non_basic_vars)

def solve_auxiliary(tableau, rows, cols):
    """Solves the auxiliary problem to find an initial feasible solution."""
    print("Problema Auxiliar:")
    tableau, non_basic, basic = identify_basis_vars(tableau, rows, cols)

    # If we already have a valid basis
    if len(non_basic) == 0:
        print("Base inicial já existe\n")
        return tableau, "otima"

    aux_tableau, aux_rows, aux_cols = build_aux_problem(tableau, rows, cols, non_basic)
    print_matrix(aux_tableau)
    print("\nExecutando Simplex no problema auxiliar:")
    aux_tableau, status = primal_phase(aux_tableau, aux_rows, aux_cols)

    # If the sum of artificial variables is not zero, no feasible solution
    if aux_tableau[0, -1] < -CONFIG.epsilon:
        return tableau, "inviavel"

    return tableau, "otima"

def simplex(tableau, rows, cols):
    """Runs the complete Simplex method (auxiliary, dual, and primal phases)."""
    # Phase 0: Solve auxiliary problem if needed
    tableau, status = solve_auxiliary(tableau, rows, cols)
    if status == "inviavel":
        return tableau, status

    # Phase 1: Dual
    tableau, status = dual_phase(tableau, rows, cols)
    if status == "inviavel":
        return tableau, status

    # Phase 2: Primal
    tableau, status = primal_phase(tableau, rows, cols)
    return tableau, status

def setup_test_env():
    """Configures the default environment for tests."""
    original = {
        "decimals": SARGS.decimals,
        "digits": SARGS.digits,
        "policy": SARGS.policy,
        "number_format": CONFIG.number_format,
    }
    SARGS.decimals = 7
    SARGS.digits = 7
    SARGS.policy = "largest"
    CONFIG.number_format = f"{{:>{SARGS.digits}.{SARGS.decimals}f}}"
    return original

def teardown_test_env(original):
    """Restores the original environment after tests."""
    SARGS.decimals = original["decimals"]
    SARGS.digits = original["digits"]
    SARGS.policy = original["policy"]
    CONFIG.number_format = original["number_format"]

def run_test_case(test_file, examples_dir, out_dir, verbose=False):
    """Runs a single test and returns the result (None if passed)."""
    input_path = os.path.join(examples_dir, test_file)
    expected_output_path = os.path.join(out_dir, test_file)
    SARGS.filename = input_path

    try:
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Main execution
        num_rows, num_cols, objective, constraints, var_types = parse_input()
        tab, num_rows, num_cols = initialize_tableau(num_rows, num_cols, objective, constraints)
        tab, status = simplex(tab, num_rows, num_cols)

        # Format final output
        sys.stdout = io.StringIO()
        if status in ["inviavel", "ilimitada"]:
            print(status.lower())
        else:
            print("otima")
            print(CONFIG.number_format.format(tab[0, -1]))
            solution = get_primal_solution(tab, num_rows, num_cols, var_types)
            print(" ".join(CONFIG.number_format.format(x) for x in solution) + " ")
            dual_sol = get_shadow_prices(tab, num_rows)
            print(" ".join(CONFIG.number_format.format(x) for x in dual_sol))

        sys_stdout = sys.stdout.getvalue().strip()
        sys.stdout = sys.__stdout__

        # Compare with expected
        with open(expected_output_path, "r") as fexp:
            expected_output = fexp.read().strip()

        # For infeasible/unbounded, compare only the first line
        if status in ["inviavel", "ilimitada"]:
            test_passed = (sys_stdout.split("\n")[0] == expected_output.split("\n")[0])
        else:
            test_passed = (sys_stdout == expected_output)

        if test_passed:
            print(f"✓ Teste {test_file} passou!")
            if verbose:
                print("\nSaída Gerada:")
                print(sys_stdout)
                print("\nSaída Esperada:")
                print(expected_output)
        else:
            print(f"✗ Teste {test_file} falhou!")
            return {"file": test_file, "generated": sys_stdout, "expected": expected_output}

    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"✗ Teste {test_file} falhou com erro: {str(e)}")
        traceback.print_exc()
        return {"file": test_file, "error": str(e)}

    return None

def run_all_tests():
    """Runs all test files, comparing with reference outputs."""
    args = parse_cli_args()
    original_env = setup_test_env()

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        examples_dir = os.path.join(base_dir, "exemplos")
        out_dir = os.path.join(base_dir, "out")
        os.makedirs(examples_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        test_files = sorted([f for f in os.listdir(examples_dir) if not f.startswith('.')])
        failed = []
        print(f"Encontrados {len(test_files)} arquivos de teste em {examples_dir}")

        for test_file in test_files:
            result = run_test_case(test_file, examples_dir, out_dir, verbose=args.verbose)
            if result:
                failed.append(result)

        print(f"\n{'='*50}")
        print(f"Resumo dos Testes: {len(test_files) - len(failed)}/{len(test_files)} testes passaram")

        if failed:
            print("\nTestes que Falharam:")
            for test in failed:
                print("\n" + "-"*50)
                print(f"Arquivo: {test['file']}")
                if "error" in test:
                    print(f"Erro: {test['error']}")
                else:
                    print("\nSaída Gerada:")
                    print(test["generated"])
                    print("\nSaída Esperada:")
                    print(test["expected"])
    finally:
        teardown_test_env(original_env)

def setup_simplex(args):
    """Applies the parsed arguments to configure the Simplex."""
    SARGS.filename = args.filename
    SARGS.decimals = args.decimals
    SARGS.digits = args.digits
    SARGS.policy = args.policy
    CONFIG.number_format = f"{{:>{SARGS.digits}.{SARGS.decimals}f}}"

def run_simplex_method():
    """Executes the Simplex with the given parameters."""
    try:
        # Read and process input
        num_rows, num_cols, objective, constraints, var_types = parse_input()
        tableau, num_rows, num_cols = initialize_tableau(num_rows, num_cols, objective, constraints)
        tableau, status = simplex(tableau, num_rows, num_cols)

        # Print results
        print(f"Status da Solução: {status}")
        if status == "otima":
            print("\nTableau Final:")
            print_matrix(tableau)
            print()
            print(f"Valor Objetivo: {CONFIG.number_format.format(tableau[0, -1])}")

            print("Solução Primal:")
            solution = get_primal_solution(tableau, num_rows, num_cols, var_types)
            print(" ".join(CONFIG.number_format.format(x) for x in solution))

            print("Solução Dual:")
            dual_sol = get_shadow_prices(tableau, num_rows)
            print(" ".join(CONFIG.number_format.format(x) for x in dual_sol))
        else:
            print(f"Solução: {status.capitalize()}")

    except Exception as e:
        print(f"Erro durante a execução do Simplex: {str(e)}")
        traceback.print_exc()

def main():
    """Main function to handle either tests or direct Simplex execution."""
    args = parse_cli_args()
    if args.test:
        run_all_tests()
    else:
        setup_simplex(args)
        run_simplex_method()

if __name__ == "__main__":
    main()
