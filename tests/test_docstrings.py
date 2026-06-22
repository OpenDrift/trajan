"""
Test that all public methods in Traj subclasses have docstrings, and that
no subclass overrides a docstring that differs from the base class.
"""
import inspect
import pytest

from trajan.traj import Traj
from trajan.traj1d import Traj1d
from trajan.traj2d import Traj2d
from trajan.ragged import ContiguousRagged

SUBCLASSES = [Traj1d, Traj2d, ContiguousRagged]

# Methods on Traj that have docstrings (the source of truth)
BASE_METHODS = {
    name: m
    for name, m in inspect.getmembers(Traj, predicate=inspect.isfunction)
    if not name.startswith('_') and getattr(m, '__doc__', None)
}


@pytest.mark.parametrize("cls", SUBCLASSES, ids=lambda c: c.__name__)
def test_no_overriding_docstrings(cls):
    """Subclass methods defined in the class body must not carry their own docstring
    when the base class already has one — use inherit_docstrings instead."""
    offenders = []
    for name, base_method in BASE_METHODS.items():
        own = cls.__dict__.get(name)
        if own is None:
            continue  # not overridden, fine
        own_doc = getattr(own, '__doc__', None)
        if own_doc and own_doc != base_method.__doc__:
            offenders.append(
                f"{cls.__name__}.{name}: has own docstring differing from Traj.{name}"
            )
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("cls", SUBCLASSES, ids=lambda c: c.__name__)
def test_all_public_methods_have_docstrings(cls):
    """Every public method on a subclass (including inherited) must have a docstring."""
    missing = []
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith('_'):
            continue
        if not getattr(method, '__doc__', None):
            missing.append(f"{cls.__name__}.{name}")
    assert not missing, "Missing docstrings:\n" + "\n".join(f"  {m}" for m in missing)
