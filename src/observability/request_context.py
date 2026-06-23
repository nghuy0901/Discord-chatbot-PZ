from dataclasses import dataclass, field
import time
import uuid


@dataclass
class RequestContext:
    query_id: str
    request_id: str
    channel_id: str
    user_id: str
    source: str
    started_at: float = field(default_factory=time.time)

    @classmethod
    def new(cls, channel_id: str, user_id: str, source: str) -> "RequestContext":
        return cls(
            query_id=str(uuid.uuid4())[:8],
            request_id=str(uuid.uuid4()),
            channel_id=channel_id,
            user_id=user_id,
            source=source,
        )
