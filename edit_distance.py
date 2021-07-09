"""
"Multispace" (c) by Ignacio Slater M.
"Multispace" is licensed under a
Creative Commons Attribution 4.0 International License.
You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by/4.0/>.
"""
import numpy
from numpy import zeros


def needleman_wunsch_score(x: str, y: str):
    """
    Implementation of the Needleman-Wunsch algorithm ("A general method applicable to the search for
    similarities in the amino acid sequence of two proteins", 1970) to align character sequences.

    :return:
        the last row of the result matrix.
    """
    len_x, len_y = len(x), len(y)
    score = zeros((2, max(len_x, len_y)))
    for j in range(1, len_y):
        score[0][j] = score[0][j - 1] + 1
    for i in range(0, len_x):
        score[1][0] = score[0][0] + 1
        for j in range(1, len_y):
            score_sub = score[0][j - 1] + 1
            score_del = score[0][j] + 1
            score_ins = score[1][j - 1] + 1
            score[1][j] = max(score_ins, score_del, score_sub)
        score[0] = numpy.copy(score[1])
    return numpy.copy(score[1])
