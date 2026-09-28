"""
Declarative Base for all ORM models.

Deliberately contains nothing but Base itself. Model registration lives in
app/models/__init__.py, not here -- importing model modules from this file
created a circular import (models import Base from here; if something
imported a model submodule directly before anything had imported this
module, Python would re-enter this file mid-import and fail on whichever
model was still being defined). Base has no reason to know its models exist;
the models package is the right owner of "these are all the models."
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
