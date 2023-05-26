from ast import Import
from typing import Callable, Dict, List, Final, Optional, Tuple
import time
import copy
import sympy
from sympy import symbols
from sympy.core import Expr
from sympy.core import Symbol
import numpy
from numpy import ndarray

# from scipy.ndimage._interpolation import shift
from numpy.ma import MaskedArray


class PDEFunction(object):
    """
    利用sympy对表达式进行化简

    @deprecated 由于速度太慢，不建议使用
    """

    __expr: Expr
    # 用于存储输入的expr
    __symbols: Dict[chr, Symbol]
    # 用于存储输入的符号对应表
    __centre: Final[Symbol] = symbols("centre")
    # 创建一个变量，用于代表中间的待计算的数值

    def __init__(self, expr: Expr, **symbols: Symbol) -> None:
        """
        构建表达式，以供计算

        @param expr 待计算的表达式
        @param symbols 符号对应表，a,b,c,d,e分别代表
        $\frac{\partial^2 U}{\partial x^2}$,
        $\frac{\partial^2 U}{\partial x \partial y}$,
        $\frac{\partial^2 U}{\partial y^2}$,
        $\frac{\partial U}{\partial x}$,
        $\frac{\partial U}{\partial y}$
        """
        self.__expr = expr
        self.__symbols = symbols

    def getExpr(self) -> Expr:
        """
        返回已保存的表达式

        @return 已保存的表达式
        """
        return self.__expr

    def getSymbols(self) -> Dict[str, Symbol]:
        """
        返回已保存的符号列表

        @return 已保存的符号列表
        """
        return self.__symbols

    def solve(
        self,
        canvas: List[List[float]],
        mask: List[List[bool]],
        xBegin: float = 0,
        yBegin: float = 0,
        dealta: float = 1,
        default: float = 0,
        maxIter: int = 100,
    ) -> List[List[float]]:
        """
        对函数进行数值求解

        @param canvas 初始值，并且限定画布范围
        @param mask 掩码，用于确定画布上的那些值是固定不变的，标记为True
        @param xBegin 确定画布上[0][0]的点代表的x的值
        @param yBegin 确定画布上[0][0]的点代表的y的值
        @param delta 各相邻点之间所代表的x或y值之差
        @param default 画布边界点的默认值
        @param maxIter 迭代次数

        @return 计算结果画布
        """

        def __getValue(
            self,
            martix: List[List[float]],
            i: int,
            j: int,
            default: float = 0,
        ) -> List[float]:
            """
            用于获取给定点的周围数值，若无法获取，值为default

            @param martix 源数据
            @param i 给定点在martix中的i坐标
            @param j 给定点在martix中的j坐标
            @param default 当值无法获取时的默认值

            @return 代表martix[i-1][j-1],martix[i-1][j],martix[i-1][j+1],...,martix[i+1][j+1]值的列表
            """
            params: List[float] = list()
            for m in (-1, 0, 1):
                for n in (-1, 0, 1):
                    result: float
                    try:
                        result = martix[i + m][j + n]
                    except:
                        if default == None:
                            result = martix[i][j]
                        else:
                            result = default
                    params.append(result)
            return params

        # bufTime = time.time()
        result = copy.deepcopy(canvas)
        # print(str(0) + " " + str(time.time() - bufTime))
        for n in range(maxIter):
            for i in range(len(canvas)):
                for j in range(len(canvas[i])):
                    if mask[i][j]:
                        # 跳过被掩码遮盖的单元
                        continue
                    else:
                        # 将偏导数用近似值取代
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
                        # 将x,y值代入
                        if "x" in keys:
                            rules.update({self.__symbols["x"]: xBegin + i * dealta})
                        if "y" in keys:
                            rules.update({self.__symbols["y"]: yBegin + j * dealta})
                        # 应用规则
                        eq = self.__expr.subs(rules)
                        # 解出值
                        result[i][j] = sympy.solve(eq, self.__centre)[0].evalf(3)
            # print(str(n + 1) + " " + str(time.time() - bufTime))
            # bufTime = time.time()
        return result


class TestPDEFunction(object):
    """
    利用numpy对表达式进行并行计算
    函数应先化简为标准形式，并给出各个系数的表达式
    $$
    A\frac{\partial^2 U}{\partial x^2}+
    B\frac{\partial^2 U}{\partial x \partial y}+
    C\frac{\partial^2 U}{\partial y^2}+
    D\frac{\partial U}{\partial x}+
    E\frac{\partial U}{\partial y}=
    F
    $$

    @preview 不稳定版本
    """

    def __init__(
        self,
        A: Callable[[float, float], float] = lambda x, y: 0,
        B: Callable[[float, float], float] = lambda x, y: 0,
        C: Callable[[float, float], float] = lambda x, y: 0,
        D: Callable[[float, float], float] = lambda x, y: 0,
        E: Callable[[float, float], float] = lambda x, y: 0,
        F: Callable[[float, float], float] = lambda x, y: 0,
    ) -> None:
        self.__args: Final[Tuple[Callable[[float, float], float]]] = (A, B, C, D, E, F)

    def getArgs(self) -> Tuple[Callable[[float, float], float]]:
        return self.__args

    def solve(
        self,
        canvas: ndarray[ndarray[float]],
        mask: ndarray[ndarray[bool]],
        xBegin: float = 0,
        yBegin: float = 0,
        dealta: float = 1,
        maxIter: int = 100,
    ) -> ndarray[ndarray[float]]:
        def shift(
            martix: ndarray[ndarray[float]], shift: tuple[int, int], cval: float = 0
        ):
            result = numpy.empty(shape=(martix.shape[0], martix.shape[1]), dtype=float)
            result.fill(cval)
            if shift[0] > 0 and shift[1] > 0:
                result[shift[0] :, shift[1] :] = martix[0 : -shift[0], 0 : -shift[1]]
            elif shift[0] > 0 and shift[1] < 0:
                result[shift[0] :, 0 : shift[1]] = martix[0 : -shift[0], -shift[1] :]
            elif shift[0] < 0 and shift[1] > 0:
                result[0 : shift[0], shift[1] :] = martix[-shift[0] :, 0 : -shift[1]]
            elif shift[0] < 0 and shift[1] < 0:
                result[0 : shift[0], 0 : shift[1]] = martix[-shift[0] :, -shift[1] :]
            elif shift[0] == 0 and shift[1] > 0:
                result[:, shift[1] :] = martix[:, 0 : -shift[1]]
            elif shift[0] == 0 and shift[1] < 0:
                result[:, 0 : shift[1]] = martix[:, -shift[1] :]
            elif shift[0] > 0 and shift[1] == 0:
                result[shift[0] :, :] = martix[0 : -shift[0], :]
            elif shift[0] < 0 and shift[1] == 0:
                result[0 : shift[0], :] = martix[-shift[0] :, :]
            elif shift[0] == 0 and shift[1] == 0:
                result = martix
            return result

        def getRatio(
            x: ndarray[ndarray[float]],
            y: ndarray[ndarray[float]],
        ) -> tuple[ndarray[ndarray[ndarray[ndarray[float]]]], ndarray[ndarray[float]]]:
            localArgs = []
            for i in range(6):
                localArgs.append(args[i](x, y))
            for i in range(6):
                if type(localArgs[i]) != ndarray:
                    tmp = numpy.empty(shape=canvas.shape, dtype=float)
                    tmp.fill(localArgs[i])
                    localArgs[i] = tmp
            result = (
                numpy.array(
                    object=[
                        [
                            localArgs[1] / (4 * dealta**2),
                            localArgs[0] / dealta**2 - localArgs[3] / (2 * dealta),
                            -localArgs[1] / (4 * dealta**2),
                        ],
                        [
                            localArgs[2] / dealta**2 - localArgs[4] / (2 * dealta),
                            -2 * (localArgs[0] + localArgs[2]) / dealta**2,
                            localArgs[2] / dealta**2 + localArgs[4] / (2 * dealta),
                        ],
                        [
                            -localArgs[1] / (4 * dealta**2),
                            localArgs[0] / dealta**2 + localArgs[3] / (2 * dealta),
                            localArgs[1] / (4 * dealta**2),
                        ],
                    ],
                    dtype=float,
                ),
                localArgs[5],
            )
            return result

        def getValue(
            ratio: ndarray[ndarray[ndarray[ndarray[float]]]],
            canvas: ndarray[ndarray[float]],
        ) -> ndarray[ndarray[ndarray[ndarray[float]]]]:
            result = numpy.empty(
                shape=(3, 3, canvas.shape[0], canvas.shape[1]), dtype=float
            )
            for i in range(3):
                for j in range(3):
                    result[i, j] = ratio[i, j] * shift(
                        canvas, (1 - i, 1 - j), cval=numpy.nan
                    )
            return result

        def getResult(
            value: ndarray[ndarray[ndarray[ndarray[float]]]],
            ratio: ndarray[ndarray[ndarray[ndarray[float]]]],
            ratioF: ndarray[ndarray[float]],
        ) -> ndarray[ndarray[ndarray[ndarray[float]]]]:
            result = numpy.empty(
                shape=(3, 3, canvas.shape[0], canvas.shape[1]), dtype=float
            )
            sum = ratioF - value.sum(axis=(0, 1))
            localRatio = copy.deepcopy(ratio)
            localRatio[ratio == 0] = numpy.nan
            for i in range(3):
                for j in range(3):
                    result[i][j] = (sum + value[i][j]) / localRatio[i][j]
            return result

        localCanvas = copy.deepcopy(canvas)
        args = self.__args
        x = numpy.linspace(
            xBegin, xBegin + dealta * (canvas.shape[0] - 1), canvas.shape[0]
        )
        y = numpy.linspace(
            yBegin, yBegin + dealta * (canvas.shape[1] - 1), canvas.shape[1]
        )
        Y, X = numpy.meshgrid(y, x)
        ratio, ratioF = getRatio(X, Y)
        for _ in range(maxIter):
            value = getValue(ratio, localCanvas)
            result = getResult(value, ratio, ratioF)
            for i in range(3):
                for j in range(3):
                    result[i][j] = shift(result[i][j], (1 - i, 1 - j), cval=numpy.nan)
            maskedResult = MaskedArray(
                data=result,
                mask=numpy.any([numpy.isnan(result), numpy.isinf(result)], axis=0),
                dtype=float,
            )
            agResult: MaskedArray = MaskedArray(
                [
                    maskedResult.data[0, 0],
                    maskedResult.data[0, 2],
                    maskedResult.data[2, 0],
                    maskedResult.data[2, 2],
                ],
                [
                    maskedResult.mask[0, 0],
                    maskedResult.mask[0, 2],
                    maskedResult.mask[2, 0],
                    maskedResult.mask[2, 2],
                ],
            ).mean(axis=0, dtype=float)
            # for _ in range(0):
            #    rtData = numpy.append(rtData, maskedResult.data[0, 0])
            #    rtData = numpy.append(rtData, maskedResult.data[0, 2])
            #    rtData = numpy.append(rtData, maskedResult.data[2, 0])
            #    rtData = numpy.append(rtData, maskedResult.data[2, 2])
            #    rtMask = numpy.append(rtMask, maskedResult.mask[0, 0])
            #    rtMask = numpy.append(rtMask, maskedResult.mask[0, 2])
            #    rtMask = numpy.append(rtMask, maskedResult.mask[2, 0])
            #    rtMask = numpy.append(rtMask, maskedResult.mask[2, 2])
            sdResult: MaskedArray = MaskedArray(
                [
                    maskedResult.data[0, 1],
                    maskedResult.data[1, 0],
                    maskedResult.data[1, 2],
                    maskedResult.data[2, 1],
                ],
                [
                    maskedResult.mask[0, 1],
                    maskedResult.mask[1, 0],
                    maskedResult.mask[1, 2],
                    maskedResult.mask[2, 1],
                ],
            ).mean(axis=0, dtype=float)
            # for _ in range(0):
            #    rtData = numpy.append(rtData, maskedResult.data[0, 1])
            #    rtData = numpy.append(rtData, maskedResult.data[1, 0])
            #    rtData = numpy.append(rtData, maskedResult.data[1, 2])
            #    rtData = numpy.append(rtData, maskedResult.data[2, 1])
            #    rtMask = numpy.append(rtMask, maskedResult.mask[0, 1])
            #    rtMask = numpy.append(rtMask, maskedResult.mask[1, 0])
            #    rtMask = numpy.append(rtMask, maskedResult.mask[1, 2])
            #    rtMask = numpy.append(rtMask, maskedResult.mask[2, 1])
            ctResult: MaskedArray = maskedResult[1, 1]
            # for _ in range(9):
            #    rtData = numpy.append(rtData, maskedResult.data[1, 1])
            #    rtMask = numpy.append(rtMask, maskedResult.mask[1, 1])
            iterResult = ctResult
            iterResult[iterResult.mask] = sdResult[iterResult.mask]
            iterResult[iterResult.mask] = agResult[iterResult.mask]
            #rtData = rtData.reshape((9, canvas.shape[0], canvas.shape[1]))
            #rtMask = rtMask.reshape((9, canvas.shape[0], canvas.shape[1]))
            #rtResult = MaskedArray(rtData, rtMask)
            #iterResult: MaskedArray = rtResult.mean(axis=0)
            #errNum: ndarray = iterResult.mask
            iterResult[iterResult.mask] = localCanvas[iterResult.mask]
            localCanvas[mask == False] = iterResult.data[mask == False]
        return localCanvas
