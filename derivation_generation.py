from sympy import init_printing
from tqdm.auto import tqdm
import numpy as np
import sympy as sp
import pandas as pd
import random
import string
import json
from func_timeout import func_set_timeout, FunctionTimedOut
#from call_function_with_timeout import SetTimeoutDecorator
init_printing()

symbol = lambda x: sp.Symbol(x, commutative = True)

identifiers = [
    'S','\\mathbf{E}','Q','\\varepsilon_0','\\mathbf{A}','V','Q','\\rho',
    '\\nabla','\\mathbf{D}','\\mathbf{P}','\\rho_b','\\rho_f','\\mathbf{r}','\\pi',
    '\\mathbf{s}','\\delta','\\phi','\\phi_1','\\phi_2','r','\\hat{\\mathbf{r}}',
    '\\theta','C_1','C_2','C','\\mathbf{g}','m','G','\\mathbf{S}','\\varphi',
    '\\varepsilon','u','v','x','y','\\mathbf{F}','q','\\mathbf{B}','\\mathbf{v}',
    '\\mathbf{f}','\\mathbf{A}','U','V_{\\mathbf{E}}','V_{\\mathbf{B}}','\\dot{\\mathbf{r}}',
    'V','L','T','\\dot{x}','\\dot{y}','\\dot{z}','A_x','A_y','A_z','t','\\ddot{x}',
    'E_x','F_x','\\hat{\\mathbf{x}}','\\mu_0','\\mathbf{J}','c_0','\mathbf{H}',
    '\\mathbf{J}_f','\\mathbf{M}','\\mathbf{J}_M','\\mathbf{P}','\\mathbf{J}_P',
    '\\sigma_x','p','\\psi','f','f^*','\\psi^*','\\hbar','i','\\tilde{g}','g',
    '\\chi','I','b','\\sigma_p','\\varphi^*','\\tilde{g}^*','z','z^*','\\omega',
    'k','a','a^{\\dagger}','\\hat{x}','\\hat{p}','E','H','\\hat{H}','A','B',
    '\\hat{x}_0','\\hat{p}_0','t_1','t_2','A_1','A_2','\\Psi','\\Omega','P_e','P_g',
    '\\hat{X}','\\mathbb{I}','x^\\prime','\\Psi^{\\dagger}','\\Psi_{\\lambda}',
    '\\hat{H}_{\\lambda}','\\lambda','E_{\\lambda}','Z','\\hat{H}_l','\\mu','l',
    '\\Psi_{nl}','E_n','n','\\eta','g_{\\varepsilon}','L_{\\varepsilon}','g^{\\prime}_{\\varepsilon}',
    '\\eta^{\\prime}','J','J_{\\varepsilon}','f^{\\prime}','y^{\\prime}','s','c',
    'F_H','F_N','h','W','r_0','M','C_d','v_t','F_c','F_g','M_E','m_s','T','v_1','v_2',
    '\\theta_1','\\theta_2','n_1','n_2','f_E','\\mathbf{p}','f_{\\mathbf{p}}',
    'f_{\\mathbf{v}}','\\mathbf{v}','v_x','v_y','v_z'
]

identifiers = list(dict.fromkeys(identifiers))

symbols = [symbol(i) for i in identifiers]

@func_set_timeout(5)
def add(x, y):
    if type(x) == sp.Equality:
        return sp.Eq(x.args[0] + y, x.args[1] + y)
    else:
        return x + y

@func_set_timeout(5)
def minus(x, y):
    if type(x) == sp.Equality:
        return sp.Eq(x.args[0] - y, x.args[1] - y)
    else:
        return x - y

@func_set_timeout(5)
def times(x, y):
    if type(x) == sp.Equality:
        return sp.Eq(x.args[0]*y, x.args[1]*y)
    else:
        return x*y

@func_set_timeout(5)
def power(x, y):
    if type(x) == sp.Equality:
        return sp.Eq(x.args[0]**y, x.args[1]**y)
    else:
        return x**y

@func_set_timeout(5)
def divide(x, y):
    if type(x) == sp.Equality:
        return sp.Eq(x.args[0]/y, x.args[1]/y)
    else:
        return x/y

@func_set_timeout(5)
def differentiate(x, y):
    if type(x) == sp.Equality:
        return sp.Eq(sp.diff(x.args[0], y, evaluate=False), sp.diff(x.args[1], y, evaluate=False))
    else:
        y = random.choice(list(x.free_symbols))
        return sp.diff(x, y, evaluate = False)

@func_set_timeout(5)
def integrate(x, y):
    if type(x) == sp.Equality:
        return sp.Eq(sp.Integral(x.args[0], y), sp.Integral(x.args[1], y))
    else:
        y = random.choice(list(x.free_symbols))
        return sp.Integral(x, y)


@func_set_timeout(5)
def evaluate_integrals(x, D):
    # assumes derivation D is a global list of equations
    # D_local is a local copy of D which is edited within this function
    # assumes a list of symbols
    # does not integrate terms containing DOUBLE INTEGRALS or higher
    
    if not contains_double_integral(x):
        #D_local = [i[0] for i in D.copy()]
        used_symbols = []
        
        for eq in D: #D_local:
            used_symbols.extend(list(eq.free_symbols))
        integral_constants = [i for i in symbols if i not in used_symbols]
        
        if type(x) == sp.Integral:
            return x.doit() + random.choice(integral_constants)
        
        elif type(x) == sp.Equality:
            elems = get_equation_elements(x)
            integrals = [i for i in elems if type(i) == sp.Integral]
            subs = [(i, i.doit() + random.choice(integral_constants)) if str(sp.simplify(i)) != str(sp.simplify(i.doit())) else (i, i) for i in integrals]
            for i in subs:
                x = x.subs(i[0], i[1])
            return x
        else:
            return x
    else:
        return x

@func_set_timeout(5)
def evaluate_derivatives(x):
    
    if type(x) == sp.Derivative:
        return x.doit()
    
    elif type(x) == sp.Equality:
        elems = get_equation_elements(x)
        derivatives = [i for i in elems if (type(i) == sp.Derivative) and ('\int' not in sp.latex(i))]
        subs = [(i, i.doit()) for i in derivatives]
        for i in subs:
            x = x.subs(i[0], i[1])
        return x
    else:
        return x

@func_set_timeout(5)
def cos(x):
    if type(x) == sp.Equality:
        return sp.Eq(sp.cos(x.args[0]), sp.cos(x.args[1]))
    else:
        return sp.cos(x)

@func_set_timeout(5)
def sin(x):
    if type(x) == sp.Equality:
        return sp.Eq(sp.sin(x.args[0]), sp.sin(x.args[1]))
    else:
        return sp.sin(x)

@func_set_timeout(5)
def exp(x):
    if type(x) == sp.Equality:
        return sp.Eq(sp.exp(x.args[0]), sp.exp(x.args[1]))
    else:
        return sp.exp(x)

@func_set_timeout(5)
def log(x):
    if type(x) == sp.Equality:
        return sp.Eq(sp.log(x.args[0]), sp.log(x.args[1]))
    else:
        return sp.log(x)

@func_set_timeout(5)
def expand(x):
    return sp.expand(x)

@func_set_timeout(5)
def substitute_LHS_for_RHS(eq_1, eq_2):
    return eq_1.subs(eq_2.args[0], eq_2.args[1])

@func_set_timeout(5)
def substitute_RHS_for_LHS(eq_1, eq_2):
    return eq_1.subs(eq_2.args[1], eq_2.args[0])


@func_set_timeout(5)
def get_premise(symbols):
    
    rules_1 = [cos, sin, exp, log]
    rules_2 = [add, minus, times, power, divide, differentiate, integrate]
    
    arity = random.choice([1,2])

    if arity == 1:
        rule = random.choice(rules_1)
        sym = random.choice(symbols)
        RHS = rule(sym)
        LHS = random.choice([i for i in symbols if i != sym])

    elif arity == 2:
        rule = random.choice([i for i in rules_2 if i not in [differentiate, integrate]])
        sym_1 = random.choice(symbols)
        sym_2 = random.choice([i for i in symbols if i != sym_1])
        RHS = rule(sym_1, sym_2)
        LHS = random.choice([i for i in symbols if i not in [sym_1, sym_2]])
        
    # make RHS more complex
    complexity = random.choice(range(2))

    for i in range(complexity):
        
        arity = random.choice([1,2])
        
        if arity == 1:
            rule = random.choice(rules_1)
            RHS = rule(RHS)
            
        elif arity == 2:
            rule = random.choice(rules_2)
            sym = random.choice(symbols)
            
            RHS = rule(RHS, sym)
            
    LHS = sp.Function(str(LHS))(*tuple(RHS.free_symbols))

    eq = sp.Eq(LHS, RHS)
    
    return eq



def renaming_premise(symbols, D):
    count = 0
    while True:
        elems = []
        for eq in D:
            elems.extend(get_equation_elements(eq))
        elems = list(dict.fromkeys(elems))

        # more than one free symbol in RHS
        elems = [i for i in elems if len(list(i.free_symbols)) > 0 and len(str(i)) > 1]

        RHS = random.choice(elems)
        LHS = sp.Function(random.choice([i for i in symbols if i not in RHS.free_symbols]))(*tuple(RHS.free_symbols))
        
        count += 1
        if count >= 100:
            return False
        
        if (type(type(RHS)) is not sp.function.UndefinedFunction and str(LHS) not in str(D)):
            break
    return sp.Eq(LHS, RHS)


@func_set_timeout(5)
def get_equation_elements(eq):
    args = list(eq.args)
    #count = 0
    while True:
        old_length = len(args)
        for i in args:
            args.extend(i.args)
        args = list(dict.fromkeys(args))
        new_length = len(args)

        #count += 1
        #if count >= 100:
         #   return False
        
        if (new_length == old_length):
            break
    return [i for i in args if type(i) != sp.Tuple]


contains_double_integral = lambda eq: True if True in ['iint' in sp.latex(i) for i in get_equation_elements(eq)] else False

@func_set_timeout(5)
def pref_eqs(D, p):
    # makes the last equation exist p times in the list
    # equation n-i exists p-i times

    out = []
    new_D = D.copy()[-p:]
    new_D.reverse()
    for i in range(len(new_D)):
        if p-1 > i:
            out.extend([new_D[i]]*(p-i-1)**3)
        
    out.reverse()
    return D + out

@func_set_timeout(5)
def valid_substitutions(rule_name, D):

    sub_options = []
    
    for i in D:
        for j in D:
            if str(i) != str(j):
                LHS, RHS = j.args
                
                if (type(LHS) is not sp.numbers.One) and (type(LHS) is not sp.numbers.Zero) and (type(RHS) is not sp.numbers.Zero) and (type(RHS) is not sp.numbers.One):
                
                    if rule_name == "substitute_LHS_for_RHS":

                        if (str(LHS) in str(i)) and (type(i.subs(LHS, RHS)) is sp.Equality) and (i.subs(LHS, RHS) != i):

                            pair = [D.index(i), D.index(j)]
                            sub_options.append(pair) if pair not in sub_options else 0

                    elif rule_name == "substitute_RHS_for_LHS":

                        if (str(RHS) in str(i)) and (type(i.subs(RHS, LHS)) is sp.Equality) and (i.subs(RHS, LHS) != i):

                            pair = [D.index(i), D.index(j)]
                            sub_options.append(pair) if (pair not in sub_options) and (pair[0] != pair[1]) else 0
                    
    return sub_options


def step(D, p_history=10, p_arity_0=5, p_renaming=1, p_arity_1=50, p_evaluate=50, p_arity_2=100, p_int_or_diff=1, p_subs=5):
    
    
    # p_history: equation n-i is p-i times more likely (prioritises more recent equations like memory)
    # p_arity_0: overall probability multiplier for arity 0 functions
    # p_renaming: relevant prob multiplier for renaming premises
    # p_arity_1: overall probability multiplier for arity 1 functions
    # p_evaluate: relative probabilty of int or diff evaluations is multiplied by p_evaluate for arity 1 functions
    # p_arity_2: overall probability multiplier for arity 2 functions
    # p_int_or_diff: relative probability multiplier for int or diff compared to other arity 2 functions without multipliers
    # p_subs: same as p_int_or_diff but for substitution functions
    
    A = [i[1] for i in D] # only annotations
    D = [i[0] for i in D] # only equations
    
    rules_0 = [
        
        get_premise


    ] + [renaming_premise]*p_renaming
    

    rules_1 = [
        
        cos,
        sin,
        exp,
        log,
        expand
        
    ] + [evaluate_derivatives, evaluate_integrals]*p_evaluate
    
    
    rules_2 = [
        
        add,
        minus,
        times,
        divide,
        power
    
    ] + [differentiate, integrate]*p_int_or_diff + [substitute_LHS_for_RHS, substitute_RHS_for_LHS]*p_subs


    # assumes D has at least one equation so far
    relevant_equation_elements = []

    for eq in D:
        relevant_equation_elements.extend(get_equation_elements(eq))
    relevant_equation_elements = list(dict.fromkeys(relevant_equation_elements))

    arity = random.choice([0]*p_arity_0 + [1]*p_arity_1 +[2]*p_arity_2)
    
    #elem_1, elem_2 = 0, 0 
    
    if arity == 0:
        rule = random.choice(rules_0)
        if rule.__name__ != "renaming_premise":
            eq = rule(symbols)
        else:
            eq = rule(symbols, D)
        annotation = rule.__name__

    if arity == 1:
        rule = random.choice(rules_1)

        # elem_1 can be equation
        elem_1 = random.choice(pref_eqs(D, p_history))
        
        if rule.__name__ != "evaluate_integrals":
            eq = rule(elem_1)
        else:
            eq = rule(elem_1, D)
                
        n = D.index(elem_1)
                
        annotation = [rule.__name__, n+1]
                

    if arity == 2:
        
        # no substitution rules if only 1 equation in D
        if len(D) == 1:
            
            rule = random.choice([i for i in rules_2 if 'subs' not in str(i.__name__)])
        else:
            
            rule = random.choice(rules_2)
        
        # substitution
        if ("subs" in rule.__name__):
            
            if valid_substitutions(rule.__name__, D):
                
                n_1, n_2 = random.choice(valid_substitutions(rule.__name__, D))

                elem_1, elem_2 = D[n_1], D[n_2]
                
                annotation = [rule.__name__, n_1+1, n_2+1]
            
            else:
                return False
        
        # integration or differentiation
        elif rule.__name__ in ['integrate', 'differentiate']:
            
            # elem_1 can be an equation
            elem_1 = random.choice(pref_eqs(D, p_history))

            # elem_2 can be an equation with components in elem_1
            elem_2 = random.choice([i for i in get_equation_elements(elem_1) if type(i) is sp.Symbol and type(i) is not sp.Integer])
            
            n = D.index(elem_1)
                    
            annotation = [rule.__name__, n+1, elem_2]
            
        
        # powers
        elif rule.__name__ in ['power']:
            
            # elem_1 can be an equation
            elem_1 = random.choice(pref_eqs(D, p_history))

            # elem_2 can be an equation with components in elem_1
            elem_2 = random.choice([i for i in get_equation_elements(elem_1) if type(i) is sp.Symbol or type(i) is sp.Integer])
            
            n = D.index(elem_1)
                    
            annotation = [rule.__name__, n+1, elem_2]
            
        else:

            # elem_1 can be an equation
            elem_1 = random.choice(pref_eqs(D, p_history))

            # elem_2 can be an equation element
            elem_2 = random.choice([i for i in relevant_equation_elements if type(i) is not sp.Integer])
            
            n = D.index(elem_1)
                    
            annotation = [rule.__name__, n+1, elem_2]

        eq = rule(elem_1, elem_2)
        
    if type(eq) == sp.Equality:
            
        swapped_eq = sp.Eq(eq.args[1], eq.args[0])

        if (eq in D) or ('Subs' in str(eq)) or ('Piecewise' in str(eq)) or (swapped_eq in D) or len(sp.latex(eq)) >= 350 or contains_bad_func_arguments(eq):
            return False

        # success
        else:
            return (eq, annotation)

    else:
        return False
    


def length():
    while True:
        #length = int(np.round(np.random.normal(18.7, 9.56, 1)))
        #if length > 3 and length < 50: #these numbers match real world data
        length = int(np.round(np.random.normal(7, 3, 1)))
        if length > 3 and length < 10:
            return length

def extract_derivation(D):

    idxs = []
    for i in range(len(D)):
        nums = [num for num in D[i][1][1:] if type(num) is int]
        entry = [i+1]
        entry.extend(nums)
        idxs.append(entry)

    # extract all equation idx dependencies as chains
    chains = []
    for i in range(len(idxs)):
        chain = [idxs[i]]
        for c in chain:
            deps = c[1:]
            if deps:
                for d in deps:
                    for idx in idxs:
                        if idx[0] == d:
                            if idx not in chain:
                                chain.append(idx)
        chain = sorted(chain)
        if chain not in chains:
            chains.append(chain)

    # select longest chain
    idx = np.argmax([len(i) for i in chains])
    longest_chain = chains[idx]
    
    
    # extract relevant steps in chain from core derivation 
    new_D = []
    for i in longest_chain:
        idx = i[0]-1
        new_D.append(D[idx])

    # make idx swaps
    correct_idxs = [[longest_chain[i][0], i+1] for i in range(len(longest_chain))]

    # fix the idxs from extracted derivation
    fixed_D = []
    for i in range(len(new_D)):
        step = new_D[i]
        eq, ann = step

        new_ann = []
        if type(ann) is list:
            for j in ann:
                if type(j) is int:
                    for idx in correct_idxs:
                        if j == idx[0]:
                            new_ann.append(idx[1])
                else:
                    new_ann.append(j)
        else:
            new_ann = ann
        new_step = eq, new_ann
        fixed_D.append(new_step)

    return fixed_D


def derivation(prior_derivation=None):
    
    if prior_derivation is None:
        eq = get_premise(symbols)
        D = [(eq, "premise")]
    else:
        D = prior_derivation

    L = length()
    
    count = 0
    while True:
        
        next_step = step(D)

        if next_step is False:
            count += 1
            
        if count >= 100:
            print("returned incomplete derivation")
            return None
        
        eval_ints = [i[1] for i in D if 'evaluate_integrals' in str(i[1])]

        if (next_step is not False) and (next_step not in D) and (next_step[1] not in eval_ints):
            D.append(next_step) 
            actual_derivation = extract_derivation(D)
            
            if len(actual_derivation) >= L:
                break
    return actual_derivation


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, sp.Integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def clean_derivation(d):
    # gets rid of unwanted text such as \\left / \\right
    # keeps step = (annotation, eq) format
    new_d = []
    for step in d:
        annotation, eq = step
        eq = eq.replace("\\left","").replace("\\right","").replace(' )',')')
        if type(annotation) is list:
            annotation = [elem.replace("\\left","").replace("\\right","").replace(' )',')') if "\\left" in str(elem) or "\\right" in str(elem) else elem for elem in annotation]
        new_d.append((annotation, eq))
    return new_d


def create_example(derivation):
    # from ACTUAL derivation
    # also outputs srepr for reproducing derivations
    latex_derivation = []
    srepr_derivation = []
    for step in derivation:
        eq, annotation = step
        latex_eq = sp.latex(eq)
        srepr_eq = sp.srepr(eq)
        if type(annotation) is list:
            latex_annotation = [sp.latex(elem) if is_math(elem) else elem for elem in annotation]
            srepr_annotation = [sp.srepr(elem) if is_math(elem) else elem for elem in annotation]
        else:
            latex_annotation = annotation
            srepr_annotation = annotation
        latex_derivation.append([latex_annotation, latex_eq])
        srepr_derivation.append([srepr_annotation, srepr_eq])
    latex_derivation = clean_derivation(latex_derivation)
    example = {
        "derivation":latex_derivation,
        "srepr_derivation":srepr_derivation,
    }
    return example


def contains_bad_func_arguments(eq):
    functions = [i for i in get_equation_elements(eq) if type(type(i)) is sp.function.UndefinedFunction]
    for f in functions:
        func_args = get_equation_elements(f)
        bad_list = [i for i in func_args if type(type(i)) is sp.function.UndefinedFunction]
        if bad_list:
            return True
    return False


is_math = lambda elem: str(elem) not in string.digits and type(elem) is not str


if __name__ == '__main__':

    examples = []
    T = 30000
    with tqdm(total=T) as pbar:
        while True:
            try:
                d = derivation()
                if d not in examples:
                    example = create_example(d)
                    examples.append(example)
                    with open('derivations.json', 'w') as f:
                        json.dump(examples, f, cls = NpEncoder)
                    pbar.update(1)
                if len(examples) == T:
                    break

            except:
                pass
