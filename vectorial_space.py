"""
"Multispace" (c) by Ignacio Slater M.
"Multispace" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by/4.0/>.
"""
import numpy


class VectorSpace:
    """
    An n-ary vectorial space.
    """
    __points: numpy.ndarray

    def __init__(self, n: int):
        """
        Creates a new vectorial space with `n` dimensions.
        """
        self.__points = numpy.random.rand(10000, n)
