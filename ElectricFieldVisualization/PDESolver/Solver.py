# 尽管python是弱类型语言，但是为了使得程序更清楚，我们依然使用了类型注释
from typing import Callable, Final, Union, Any
from sympy.core import Expr
from sympy.core import Symbol
from numpy import ndarray
from numpy.ma import MaskedArray

# 导入其他python库
import copy
import sympy
import numpy

# 这些是类型别名，它的意义是为了更加明确的标记数据类型，要不然全是ndarray类型注释的效果就不好了
canvas_ = ndarray
rdCanvas_ = ndarray
# Callable类型标注代表它是一个函数，接受的参数标注在其方括号内第一个方括号中，返回的参数排在之后
# Union类型标注代表它的返回类型不确定，所有可能的类型标注在了其后的方括号内
factor_ = Callable[[canvas_, canvas_], Union[canvas_, float]]


class PDEFunction(object):
    """
    利用sympy对表达式进行化简
    利用sympy对表达式进行化简
    @deprecated 由于速度太慢，不建议使用
    """

    # 以下为定义的类属性，已被修饰为私有

    __expr: Expr
    # 用于存储输入的expr
    # expr这个变量用来存储表达式
    __symbols: dict[str, Symbol]
    # 用于存储输入的符号对应表
    # symbols这个变量用于存储输入的符号对应表
    __centre: Final[Symbol] = sympy.symbols("centre")
    # 创建一个变量，用于代表中间的待计算的数值
    # 我们采用的是取一个点周围的八个点来近似计算这个点，而centre创建一个变量，用于代表中间的待计算的数值

    # 我们接着定义了一个构造一个函数，用于构建表达式，以供计算
    def __init__(self, expr: Expr, **symbols: Symbol) -> None:
        r"""
        构建表达式，以供计算
        @param expr 待计算的表达式
        @param symbols 符号对应表，a,b,c,d,e分别代表
        $\frac{\partial^2 U}{\partial x^2}$,
        $\frac{\partial^2 U}{\partial x \partial y}$,
        $\frac{\partial^2 U}{\partial y^2}$,
        $\frac{\partial U}{\partial x}$,
        $\frac{\partial U}{\partial y}$
        以及x,y
        """
        self.__expr = expr
        self.__symbols = symbols

    # 我们在该类下定义一个solve方法，对传入的函数进行数值求解
    # 该函数的参数说明已经写在了函数注释中
    def solve(
        self,
        canvas: list[list[float]],
        mask: list[list[bool]],
        xBegin: float = 0,
        yBegin: float = 0,
        dealta: float = 1,
        default: float = 0.0,
        maxIter: int = 100,
    ) -> list[list[float]]:
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

        # 我们在solve方法下嵌套定义了getvalue函数，用于获取给定点的周围的8个数值
        # 若无法获取，则使用传入的default值代替
        def getValue(
            martix: list[list[float]],
            i: int,
            j: int,
            default: float = 0.0,
        ) -> list[float]:
            """
            用于获取给定点的周围数值，若无法获取，值为default
            @param martix 源数据
            @param i 给定点在martix中的i坐标
            @param j 给定点在martix中的j坐标
            @param default 当值无法获取时的默认值
            @return 代表martix[i-1][j-1],martix[i-1][j],martix[i-1][j+1],...,martix[i+1][j+1]值的列表
            """
            params: list[float] = list()  # 用于存储这九个计算出来的值
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

        symbols = self.__symbols
        centre = self.__centre
        result = copy.deepcopy(canvas)  # 将输入的canvas拷贝一遍，以防直接对实参进行修改
        for n in range(maxIter):  # 循环迭代
            for i in range(len(canvas)):
                for j in range(len(canvas[i])):
                    # 固定的值不需要计算，未固定的值需要重新计算并且替换原来的值
                    if mask[i][j]:
                        # 跳过被掩码遮盖的单元
                        continue
                    else:
                        # 将偏导数用近似公式取代
                        params = getValue(result, i, j, default)  # 获得result[i][j]周围的值
                        # 我们用rules变量来记录替换规则，keys来确定要替换哪些符号
                        rules: dict[Symbol, Union[Expr, float]] = dict()  # 记录替换规则
                        keys = symbols.keys()  # 确定要替换哪些符号，如果有就替换，没有的话就不替换
                        if "a" in keys:
                            rules.update(
                                {
                                    symbols["a"]: (params[5] + params[3] - 2 * centre)
                                    / dealta**2
                                }
                            )
                        if "b" in keys:
                            rules.update(
                                {
                                    symbols["b"]: (params[2] + params[6] - 2 * centre)
                                    / (2 * dealta**2)
                                }
                            )
                        if "c" in keys:
                            rules.update(
                                {
                                    symbols["c"]: (params[1] + params[7] - 2 * centre)
                                    / dealta**2
                                }
                            )
                        if "d" in keys:
                            rules.update({symbols["d"]: (params[5] - centre) / dealta})
                        if "e" in keys:
                            rules.update({symbols["e"]: (params[1] - centre) / dealta})
                        # 将x,y值代入
                        # 由于abcde是关于x，y的表达式，我们也需要将下x，y替换掉
                        if "x" in keys:
                            rules.update({symbols["x"]: xBegin + i * dealta})
                        if "y" in keys:
                            rules.update({symbols["y"]: yBegin + j * dealta})
                        # 应用规则
                        eq = self.__expr.subs(rules)
                        # 解出值
                        result[i][j] = sympy.solve(eq, centre)[0].evalf(3)
        # 返回的结果存在了test1.py中的z变量中，并在最后绘出图像
        return result


class TestPDEFunction(object):
    r"""
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
    """

    __args: list[factor_]
    # 我们定义了一个类属性，它是一个列表，所有的系数表达式都存在了里面
    # 所有的系数表达式都存在这里了

    # 我们定义了一个构造函数函数，将ABCDEF这些系数放入__args变量中。
    def __init__(
        self,
        A: factor_ = lambda x, y: 0,
        B: factor_ = lambda x, y: 0,
        C: factor_ = lambda x, y: 0,
        D: factor_ = lambda x, y: 0,
        E: factor_ = lambda x, y: 0,
        F: factor_ = lambda x, y: 0,
    ) -> None:
        """
        构建表达式，以供计算
        其中，A,B,C不应该全为零
        @param A 系数A，关于x,y的表达式
        @param B 系数B，关于x,y的表达式
        @param C 系数C，关于x,y的表达式
        @param D 系数D，关于x,y的表达式
        @param E 系数E，关于x,y的表达式
        @param F 系数F，关于x,y的表达式
        """
        self.__args = [A, B, C, D, E, F]

    # 相似的，我们定义一个solve方法，对函数进行数值求解
    # 该函数的参数说明已经写在了函数注释中
    def solve(
        self,
        canvas: canvas_,
        mask: canvas_,
        xBegin: float = 0,
        yBegin: float = 0,
        dealta: float = 1,
        maxIter: int = 100,
        expansion: int = 2,
    ) -> canvas_:
        """
        对函数进行数值求解
        @param canvas 初始值，并且限定画布范围
        @param mask 掩码，用于确定画布上的那些值是固定不变的，标记为True
        @param xBegin 确定画布上[0][0]的点代表的x的值
        @param yBegin 确定画布上[0][0]的点代表的y的值
        @param delta 各相邻点之间所代表的x或y值之差
        @param maxIter 迭代次数
        @param expansion 在计算前将画布扩大，以提升精确度
        @return 计算结果画布
        """

        def expand(array: canvas_, x: int, y: int):
            # 我们为了提升精度，将画布扩大（正数）或缩小（负数）x,y倍
            # 如果x，y是正的，就执行扩大，如果是负的就执行缩小
            """
            将画布扩大（正数）或缩小（负数）x,y倍
            @param array 画布
            @param x x轴缩放比例
            @param y y轴缩放比例
            @return 结果画布
            """
            shape_x, shape_y = array.shape
            if x > 0:
                shape_x = shape_x * x
            if x < 0:
                shape_x = shape_x // (-x)
            tmp = numpy.empty((shape_x, shape_y), float)
            if x > 0:
                for i in range(array.shape[0]):
                    tmp[i * x : (i + 1) * x, :] = array[i : i + 1, :]
            if x < 0:
                for i in range(tmp.shape[0]):
                    tmp[i, :] = array[-i * x : -(i + 1) * x, :].mean(0)
            if y > 0:
                shape_y = shape_y * y
            if y < 0:
                shape_y = shape_y // (-y)
            result = numpy.empty((shape_x, shape_y), float)
            if y > 0:
                for i in range(tmp.shape[1]):
                    result[:, i * y : (i + 1) * y] = tmp[:, i : i + 1]
            if y < 0:
                for i in range(result.shape[1]):
                    result[:, i] = tmp[:, -i * y : -(i + 1) * y].mean(1)
            return result

        # 因为ABCDEF是关于x，y的函数，如果确定了x，y 那么ABCDEF是可以算出来的
        # 如果算出了ABCDEF，我们就可以将 INFO.md 3.1 公式前面的系数算出来
        # 最后我们将这些系数存放在一个四位数组之中，使用四维数组纯属是为了方便后续编码
        # 这里的四维数组并不复杂，我们也可以将他看作一个3*3的以公式顺序排列的二维数组，然后随便访问其中一个，
        # 里面存在一个和传入画布一样大小的二维数组，再给定i,j坐标，即可访问到所需的值
        def getRatio(
            x: canvas_,
            y: canvas_,
        ) -> tuple[ndarray[canvas_, Any], canvas_]:
            r"""
            获得系数的值
            什么是系数？参见INFO.md
            访问时先访问系数位置（例如$+\frac{B}{4\Delta^2}$就在[0,0]）,再访问值位置
            @param x x坐标
            @param y y坐标
            @return 所有的坐标下的系数值，以及F的值
            """
            localArgs = []
            for i in range(6):
                # self.__args中的元素都是关于x,y的函数，因此需要传入参数值x和y
                localArgs.append(self.__args[i](x, y))
            for i in range(6):
                if type(localArgs[i]) != ndarray:
                    tmp = numpy.empty(shape=localCanvas.shape, dtype=float)
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

        # getValue这个函数的功能是将获得的系数乘上旧迭代值，算出多项式 3.1 中每一项的值
        def getValue(
            ratio: ndarray[canvas_, Any],
            canvas: canvas_,
        ) -> ndarray[rdCanvas_, Any]:
            """
            获得系数乘上旧迭代值的结果
            @param ratio 所有系数值
            @param canvas 旧的迭代值
            @return 所有相乘后的值
            """
            result = numpy.empty(
                shape=(3, 3, localCanvas.shape[0] - 2, localCanvas.shape[1] - 2),
                dtype=float,
            )
            for i in range(3):
                for j in range(3):
                    result[i, j] = (
                        ratio[i, j, 1:-1, 1:-1]
                        * canvas[
                            i : i + localCanvas.shape[0] - 2,
                            j : j + localCanvas.shape[1] - 2,
                        ]
                    )
            return result

        # getResult这个函数的功能是利用之前那些函数返回的结果将某个点周围的九个迭代之后的值算出来
        def getResult(
            value: ndarray[rdCanvas_, Any],
            ratio: ndarray[canvas_, Any],
            ratioF: canvas_,
        ) -> ndarray[rdCanvas_, Any]:
            result: ndarray[rdCanvas_, Any] = numpy.empty(
                shape=(3, 3, localCanvas.shape[0] - 2, localCanvas.shape[1] - 2),
                dtype=float,
            )
            """
            获得新的迭代结果值
            @param value getValue的返回值
            @param ratio 所有系数的值
            @param ratioF 所有系数F的值
            @return 新的迭代结果
            """
            sum: rdCanvas_ = ratioF[1:-1, 1:-1] - value.sum(axis=(0, 1))
            localRatio: ndarray[rdCanvas_, Any] = copy.deepcopy(ratio[:, :, 1:-1, 1:-1])
            rdRatio: ndarray[rdCanvas_, Any] = ratio[:, :, 1:-1, 1:-1]
            localRatio[rdRatio == 0] = numpy.nan
            for i in range(3):
                for j in range(3):
                    result[i][j] = (sum + value[i][j]) / localRatio[i][j]
            return result

        # 在一切之前，我们进行了深拷贝，防止直接修改实参
        localCanvas = copy.deepcopy(canvas)  # 深拷贝，防止直接修改实参
        localCanvas = expand(localCanvas, expansion, expansion)
        localMask = copy.deepcopy(mask)
        localMask = expand(localMask, expansion, expansion)
        dealta = dealta / expansion

        # 我们需要对getradio这个函数做准备，为了确定每个画布上x，y的值
        x = numpy.linspace(
            xBegin, xBegin + dealta * (localCanvas.shape[0] - 1), localCanvas.shape[0]
        )
        y = numpy.linspace(
            yBegin, yBegin + dealta * (localCanvas.shape[1] - 1), localCanvas.shape[1]
        )
        Y, X = numpy.meshgrid(y, x)
        ratio, ratioF = getRatio(X, Y)

        for r in range(maxIter):
            if r != 0:  # 由于 r==0 时已经扩大过一次画布，故只需要在之后的循环中将画布扩大
                localCanvas = expand(localCanvas, expansion, expansion)
            # 最终我们通过getValue和getResult算出了最终迭代的值
            value = getValue(ratio, localCanvas)
            result = getResult(value, ratio, ratioF)

            # 以下部分是对同一位置的数通过不同位置产生的9个迭代后结果进行处理
            # 这些都是通过具体调试结果所得的实验性处理方案，不保证方案的正确性，甚至可能无法得到正确结果

            # 我们将算好的九个值按照一定的顺序放入带掩码ndarray数组中
            # MaskedArray构造函数的第二个参数为掩码数组
            # 被掩码数组标记为True的位置所对应的第一个参数数组所在位置的数值将会是为无效值
            tmpResult = MaskedArray(
                numpy.empty(ratio.shape, float),
                numpy.zeros(ratio.shape, bool) == False,
            )  # 带掩码ndarray数组，第二个参数为掩码，标记为True所对应的第一个参数数组为无效值

            # 以下部分是将result数组移了位置，以便于后续求均值操作
            for i in range(3):
                for j in range(3):
                    tmpResult[
                        i,
                        j,
                        i : i + localCanvas.shape[0] - 2,
                        j : j + localCanvas.shape[0] - 2,
                    ] = result[i, j]

            # 因为边界部分有巨大的误差，我们直接将边界值舍弃
            maskedResult: MaskedArray = tmpResult[:, :, 1:-1, 1:-1]
            # 将一些不正确的数据标记为无效数据，用于排除错误数据
            maskedResult.mask = maskedResult.mask | numpy.isnan(
                maskedResult.data
            )  # 排除错误数据

            # 通过观察，当其中一个数据被标记为无效时，其中心对称的另一边的数据将会对结果产生巨大影响
            # 这一部分是为了防止出此类事件的发生，将另一边的数据也用掩码盖住，标记为无效
            def maskArr(i: int, j: int) -> None:
                maskedResult.mask[i, j] = maskedResult.mask[2 - i, 2 - j] = (
                    maskedResult.mask[i, j] | maskedResult.mask[2 - i, 2 - j]
                )

            maskArr(0, 0)
            maskArr(1, 0)
            maskArr(2, 0)
            maskArr(0, 1)
            # （然而，效果不佳）

            # 再通过观察，我们发现在无法由点周围8个数据算出近似值的情况下，
            # 使用点周围上下左右的值用来求均值更为准确，而对角上的点求均值次之，
            # 若有无效值，则取中心对称的两个值求均值即可，以此类推
            # 若都不行，则直接使用迭代前的数值
            agResult = MaskedArray(
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
            ctResult: MaskedArray = maskedResult[1, 1]
            # 所以，上面这玩意你知道是对哪些值求均值的，对吧
            iterResult = ctResult
            iterResult[iterResult.mask] = sdResult[iterResult.mask]
            iterResult[iterResult.mask] = agResult[iterResult.mask]
            iterResult[iterResult.mask] = localCanvas[1:-1, 1:-1][iterResult.mask]
            # 优先顺序：ctResult>sdResult>agResult>旧值

            # 考虑传入的mask参数，防修改被标记为固定值的数据
            rdMask: rdCanvas_ = localMask[1:-1, 1:-1]
            localCanvas[1:-1, 1:-1][rdMask == False] = iterResult.data[rdMask == False]
            # 将画布恢复为原来的形状
            localCanvas = expand(localCanvas, -expansion, -expansion)
        # 最后将结果返回test2.py中的调用者，继而将图绘出
        return localCanvas
