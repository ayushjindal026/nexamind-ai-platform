"""
Auth endpoints. Thin on purpose: all business logic lives in auth/service.py,
this module only translates domain exceptions into HTTP responses.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_membership, get_current_user
from app.auth.service import (
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    authenticate_user,
    register_user,
)
from app.core.security import create_access_token
from app.db.session import get_db
from app.models.membership import Membership
from app.models.user import User
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    MeResponse,
    OrganizationOut,
    RegisterRequest,
    TokenResponse,
    UserOut,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    try:
        user, organization = register_user(
            db, payload.email, payload.password, payload.organization_name
        )
    except EmailAlreadyRegisteredError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    access_token = create_access_token(str(user.id))
    return TokenResponse(
        access_token=access_token,
        user=UserOut(id=user.id, email=user.email),
        organization=OrganizationOut(id=organization.id, name=organization.name),
    )


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    try:
        user = authenticate_user(db, payload.email, payload.password)
    except InvalidCredentialsError:
        # Deliberately identical for "no such user" and "wrong password" — see service.py.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    return LoginResponse(access_token=create_access_token(str(user.id)))


@router.get("/me", response_model=MeResponse)
def me(
    user: User = Depends(get_current_user),
    membership: Membership = Depends(get_current_membership),
) -> MeResponse:
    return MeResponse(
        id=user.id,
        email=user.email,
        organization=OrganizationOut(id=membership.organization.id, name=membership.organization.name),
        role=membership.role.value,
    )
