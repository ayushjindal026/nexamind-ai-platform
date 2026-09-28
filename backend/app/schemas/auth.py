"""
Pydantic schemas for the auth API. Deliberately small — one file, since
Phase 2's entire surface is three endpoints.
"""

import uuid

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    organization_name: str = Field(min_length=1, max_length=255)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class OrganizationOut(BaseModel):
    id: uuid.UUID
    name: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: str


class TokenResponse(BaseModel):
    """Returned by /register — includes user + org since registration auto-logs-in."""

    access_token: str
    token_type: str = "bearer"
    user: UserOut
    organization: OrganizationOut


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    id: uuid.UUID
    email: str
    organization: OrganizationOut
    role: str
