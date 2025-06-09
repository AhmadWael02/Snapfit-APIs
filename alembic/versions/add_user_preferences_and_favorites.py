"""add user preferences and favorites

Revision ID: add_user_preferences_and_favorites
Revises: previous_revision
Create Date: 2024-03-19 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_user_preferences_and_favorites'
down_revision = 'previous_revision'  # Update this to your previous migration
branch_labels = None
depends_on = None

def upgrade():
    # Create user_preferences table
    op.create_table(
        'user_preferences',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('fit_preference', sa.String(), nullable=False, server_default='Regular'),
        sa.Column('lifestyle_preferences', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('season_preference', sa.String(), nullable=False, server_default='Auto'),
        sa.Column('age_group', sa.String(), nullable=False, server_default='18-24'),
        sa.Column('preferred_colors', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('excluded_categories', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id')
    )

    # Create user_favorites table
    op.create_table(
        'user_favorites',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('item_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['item_id'], ['clothes.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_favorites_id'), 'user_favorites', ['id'], unique=False)

def downgrade():
    op.drop_index(op.f('ix_user_favorites_id'), table_name='user_favorites')
    op.drop_table('user_favorites')
    op.drop_table('user_preferences') 