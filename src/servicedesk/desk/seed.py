"""Load the service catalogue and sample tickets from data/seed/ (`servicedesk seed`).

Safe to run again: services are upserted (INSERT ... ON CONFLICT DO UPDATE), keeping any
status changed since; sample tickets are only added to an empty tickets table.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from servicedesk.db.models import Service, Ticket, TicketComment

SEED_DIR = Path("data/seed")
CATALOGUE_FIELDS = ["name", "description", "owning_team", "criticality"]


@dataclass
class SeedReport:
    services: int
    tickets_added: int


def seed(session_factory: sessionmaker[Session], seed_dir: Path = SEED_DIR) -> SeedReport:
    services = yaml.safe_load((seed_dir / "services.yaml").read_text(encoding="utf-8"))
    tickets = yaml.safe_load((seed_dir / "tickets.yaml").read_text(encoding="utf-8"))

    # A multi-row INSERT needs the same keys in every row.
    services = [{"status_message": None, **service} for service in services]

    with session_factory.begin() as session:
        stmt = insert(Service).values(services)
        # On an existing id, refresh the catalogue fields but not status/status_message:
        # re-seeding shouldn't undo a status set with `servicedesk set-status`.
        stmt = stmt.on_conflict_do_update(
            index_elements=[Service.id],
            set_={field: stmt.excluded[field] for field in CATALOGUE_FIELDS},
        )
        session.execute(stmt)

        added = 0
        if session.scalar(select(func.count()).select_from(Ticket)) == 0:
            for item in tickets:
                comments = item.pop("comments", [])
                service_id = item.pop("service", None)  # pop before **item unpacks it
                ticket = Ticket(**item, service_id=service_id)
                if ticket.status in ("resolved", "closed"):
                    ticket.resolved_at = func.now()
                ticket.comments = [TicketComment(**c) for c in comments]
                session.add(ticket)
                added += 1

    return SeedReport(services=len(services), tickets_added=added)
