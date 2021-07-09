"""
"Multispace" (c) by Ignacio Slater M.
"Multispace" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by/4.0/>.
"""
import numpy
from numpy import ndarray


class ArityMismatchException(Exception):
    def __init__(self, v_1: ndarray, v_2: ndarray):
        super(ArityMismatchException, self).__init__(
            f"Trying to operate vectors of different arities. v_1: {len(v_1)}, and v_2: {len(v_2)}")


class VectorSpace:
    """
    An n-ary vectorial space.
    """
    __arity: int
    __points: ndarray

    def __init__(self, v_arity: int, size: int):
        """
        Creates a new vectorial space.

        :param v_arity:
            the number of dimensions of the vector space
        :param size:
            the number of points on the vector space
        """
        self.__arity = v_arity
        self.__points = numpy.random.rand(size, v_arity)

    @property
    def arity(self) -> int:
        return self.__arity

    @property
    def points(self):
        return self.__points.copy()

    def __len__(self) -> int:
        return len(self.__points)


def minkowski_distance(v_1: ndarray, v_2: ndarray, p: int) -> float:
    """
    Computes the Minkowski distance of two n-ary points.

    :param v_1:
        the first point
    :param v_2:
        the second point
    :param p:
        the order of the metric
    :return:
        the distance between the points
    :raises ArityMismatchException:
        if the points have different arities
    """
    arities = (len(v_1), len(v_2))
    if arities[0] != arities[1]:
        raise ArityMismatchException(v_1, v_2)
    abs_diffs = numpy.abs(v_1 - v_2)
    p_diffs: ndarray = abs_diffs ** p
    return p_diffs.sum() ** (1 / p)


def manhattan_distance(v_1: ndarray, v_2: ndarray) -> float:
    return minkowski_distance(v_1, v_2, 1)


def max_manhattan_distance(v_space: VectorSpace) -> ndarray:
    """
    Computes the maximum manhattan distance ($L_1$) in an n-ary vector space.

    The manhattan distance for two given vectors $x$ and $y$ is defined as:
    $$
        d(x, y) = \sum_{i=1}^D |x_i - y_i|
    $$

    :return: a vector with the maximum Manhattan distance.
    """
    size = len(v_space)
    arity = v_space.arity
    maximums = numpy.zeros((size, arity))
    minimums = numpy.zeros((size, arity))
    for i in range(size):
        maximums[i] = numpy.sum(v_space.points)
        minimums[i] = numpy.subtract(v_space.points)
    maximums.sort()
    minimums.sort()
    return numpy.max(maximums[-1] - maximums[0], minimums[-1] - minimums[0])


if __name__ == '__main__':
    space = VectorSpace(v_arity=2, size=100000)
    print(minkowski_distance(space.points[0], space.points[1], 2))
