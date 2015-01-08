from helpers import arrays_similar

def test_arrays_similar():
    assert arrays_similar([],[])
    assert not arrays_similar([1],[])
    assert not arrays_similar([1],[1,2])
    assert arrays_similar([1,2],[1,2])
