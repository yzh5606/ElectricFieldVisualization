from typing import Dict, List, Final, Optional
import time
import copy
import sympy
from sympy import symbols
from sympy.core import Expr
from sympy.core import Symbol


class PDEFunction(object):
    __expr: Expr
    __symbols: Dict[chr, Symbol]

    __centre: Final[Symbol] = symbols("centre")

    def __getValue(
        self,
        martix: List[List[float]],
        i: int,
        j: int,
        default: Optional[float] = 0,
    ) -> List[float]:
        params: List[float] = list()
        for m in (-1, 0, 1):
            for n in (-1, 0, 1):
                result: float
                try:
                    result = martix[m][n]
                except:
                    if default == None:
                        result = martix[i][j]
                    else:
                        result = default
                params.append(result)
        return params

    def __init__(self, expr: Expr, **symbols: Symbol) -> None:
        self.__expr = expr
        self.__symbols = symbols

    def getExpr(self) -> Expr:
        return self.__expr

    def getSymbols(self) -> Dict[chr, Symbol]:
        return self.__symbols

    def solve(
        self,
        canvas: List[List[float]],
        mask: List[List[bool]],
        xBegin: float = 0,
        yBegin: float = 0,
        dealta: float = 1,
        default: Optional[float] = 0,
        maxIter: int = 100,
    ) -> List[List[float]]:
        bufTime = time.time()
        result = copy.deepcopy(canvas)
        print(str(0) + " " + str(time.time() - bufTime))
        for n in range(maxIter):
            for i in range(len(canvas)):
                for j in range(len(canvas[i])):
                    if mask[i][j]:
                        continue
                    else:
                        params = self.__getValue(result, i, j, default)
                        rules: Dict[Symbol, Expr] = dict()
                        keys = self.__symbols.keys()
                        if "a" in keys:
                            rules.update(
                                {
                                    self.__symbols["a"]: (
                                        params[5] + params[3] - 2 * self.__centre
                                    )
                                    / dealta**2
                                }
                            )
                        if "b" in keys:
                            rules.update(
                                {
                                    self.__symbols["b"]: (
                                        params[2] + params[6] - 2 * self.__centre
                                    )
                                    / (2 * dealta**2)
                                }
                            )
                        if "c" in keys:
                            rules.update(
                                {
                                    self.__symbols["c"]: (
                                        params[1] + params[7] - 2 * self.__centre
                                    )
                                    / dealta**2
                                }
                            )
                        if "d" in keys:
                            rules.update(
                                {
                                    self.__symbols["d"]: (params[5] - self.__centre)
                                    / dealta
                                }
                            )
                        if "e" in keys:
                            rules.update(
                                {
                                    self.__symbols["e"]: (params[1] - self.__centre)
                                    / dealta
                                }
                            )
                        if "x" in keys:
                            rules.update({self.__symbols["x"]: xBegin + i * dealta})
                        if "y" in keys:
                            rules.update({self.__symbols["y"]: yBegin + j * dealta})
                        eq = self.__expr.subs(rules)
                        result[i][j] = sympy.solve(eq, self.__centre)[0].evalf(3)
            print(str(n + 1) + " " + str(time.time() - bufTime))
            bufTime = time.time()
        return result
