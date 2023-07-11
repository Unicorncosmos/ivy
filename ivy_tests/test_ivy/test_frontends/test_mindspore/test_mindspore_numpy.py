# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

# TODO: uncomment after frontend is not required
#  to be set as backend in test_frontend_function
# @handle_frontend_test(
#     fn_tree="mindspore.numpy.array",
#     dtype_and_a=helpers.dtype_and_values(
#         available_dtypes=helpers.get_dtypes("numeric"),
#         num_arrays=1,
#         min_num_dims=0,
#         max_num_dims=5,
#         min_dim_size=1,
#         max_dim_size=5,
#     ),
#     ndmin=st.integers(min_value=0, max_value=5),
#     copy=st.booleans(),
#     test_with_out=st.just(False),
# )
# def test_mindspore_numpy_array(
#     dtype_and_a,
#     frontend,
#     test_flags,
#     fn_tree,
#     on_device,
#     copy,
#     ndmin,
# ):
#     dtype, a = dtype_and_a
#     helpers.test_frontend_function(
#         input_dtypes=dtype,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         object=a,
#         dtype=None,
#         copy=copy,
#         ndmin=ndmin,
#     )


@handle_frontend_test(
    fn_tree="mindspore.softmax",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    test_with_out=st.just(False),
)
def test_mindspore_softmax(
    *,
    dtype_x_and_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
    )
