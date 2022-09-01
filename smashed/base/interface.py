import inspect
from dataclasses import MISSING
from functools import partial
from typing import Any, Callable, Dict, Generic, Optional, Tuple, Type, TypeVar

from typing_extensions import Concatenate, ParamSpec

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


class interface(Generic[T, P, R]):
    """An interface is a decorator that select the correct method to call
    based on the types of the arguments. For example, in the class below,
    the method `add_one` is customized for the type `int` and `str`, but
    fails for any other type of `a`.

    class MyClass:
        @Interface
        def add_one(self, a: Any) -> Any:
            # fallback method
            raise TypeError(f"Type {type(a)} not supported")

        @add_one.add_interface(a=int)
        def add_one_int(self, a: int) -> int:
            # a is an int
            return a + 1

        @add_one.add_interface(a=str)
        def add_one_str(self, a: str) -> str:
            # a is a str
            return a + "1"
    """

    interfaces: Dict[Tuple[str, ...], Dict[Tuple[type, ...], Any]]

    def __init__(
        self, interfaced_method: Callable[Concatenate[Any, P], R]
    ) -> None:
        """Create an Interface object.

        Args:
            interfaced_method: The method to be interfaced; it is also the
                default method if no matching interface is found.
        """
        self.interfaces = {}
        self.interfaced_method = interfaced_method
        self.method_signature = inspect.signature(interfaced_method)

    def add_interface(
        self, **kwargs: type
    ) -> Callable[[Callable[Concatenate[Any, P], R]], "interface"]:
        """Add an interface to the Interface for specific arguments and types.

        Args:
            **kwargs: The arguments and types to add an interface for.
                the key is the argument name, the value is the type.
        """

        def _add_interface(
            method: Callable[Concatenate[Any, P], R]
        ) -> "interface":
            self.interfaces.setdefault(tuple(kwargs.keys()), {})[
                tuple(kwargs.values())
            ] = method
            return self

        return _add_interface

    def __get__(
        self, obj: Any, type: Optional[Type] = None
    ) -> Callable[Concatenate[P], R]:
        """Return a bound method that calls the correct interface."""
        return partial(self._run_interface, __obj__=obj)

    def _run_interface(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the interfaced method with the correct interface."""

        if (obj := kwargs.pop("__obj__", MISSING)) is MISSING:
            raise ValueError(
                "__obj__ is required; `Interface._run_interface` "
                "was improperly called; please file a bug report"
            )

        sig_vals = self.method_signature.bind(self, *args, **kwargs)
        method_to_call = None

        for arg_names, types_dict in self.interfaces.items():
            # create lookup key for <types, method> dictionary
            types_key = tuple(
                type(sig_vals.arguments[arg_name]) for arg_name in arg_names
            )

            # return the first one we find
            if types_key in types_dict:
                method_to_call = types_dict[types_key]
                break

        # fall back to the default method if we didn't find anything
        method_to_call = method_to_call or self.interfaced_method
        return method_to_call(obj, *args, **kwargs)
