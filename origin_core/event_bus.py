"""
Event Bus for inter-agent publish/subscribe communication.
Enables loose coupling between Origin modules through topic-based messaging.
"""

import time
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class EventPriority(Enum):
    """Priority levels for event processing order."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """Represents a single event dispatched on the bus."""
    topic: str
    payload: Dict[str, Any]
    source: str = "unknown"
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    event_id: Optional[str] = None


@dataclass
class Subscription:
    """Tracks a subscriber's callback and metadata."""
    callback: Callable[[Event], None]
    subscriber_id: str
    priority: EventPriority = EventPriority.NORMAL


class EventBus:
    """
    Lightweight publish/subscribe event bus for intra-system messaging.

    Agents, tools, and pipeline stages can publish events to named topics.
    Subscribers receive events filtered by topic, ordered by priority.

    Example:
        bus = EventBus()
        bus.subscribe("chunk.created", my_handler, subscriber_id="indexer")
        bus.publish(Event(topic="chunk.created", payload={"chunk_id": "c1"}))
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Subscription]] = defaultdict(list)
        self._event_log: List[Event] = []
        self._dead_letter: List[Event] = []

    def subscribe(
        self,
        topic: str,
        callback: Callable[[Event], None],
        subscriber_id: str = "anonymous",
        priority: EventPriority = EventPriority.NORMAL
    ) -> None:
        """Registers a callback for a specific topic."""
        sub = Subscription(
            callback=callback,
            subscriber_id=subscriber_id,
            priority=priority
        )
        self._subscribers[topic].append(sub)
        # Keep subscribers sorted by priority (highest first)
        self._subscribers[topic].sort(key=lambda s: s.priority.value, reverse=True)

    def unsubscribe(self, topic: str, subscriber_id: str) -> bool:
        """Removes a subscriber from a topic. Returns True if found."""
        subs = self._subscribers.get(topic, [])
        before = len(subs)
        self._subscribers[topic] = [s for s in subs if s.subscriber_id != subscriber_id]
        return len(self._subscribers[topic]) < before

    def publish(self, event: Event) -> int:
        """
        Dispatches an event to all subscribers of its topic.

        Returns the number of subscribers that received the event.
        """
        self._event_log.append(event)
        subs = self._subscribers.get(event.topic, [])

        if not subs:
            self._dead_letter.append(event)
            return 0

        delivered = 0
        for sub in subs:
            try:
                sub.callback(event)
                delivered += 1
            except Exception:
                # Swallow handler errors to avoid cascading failures
                pass

        return delivered

    def get_topics(self) -> List[str]:
        """Returns all topics with active subscribers."""
        return list(self._subscribers.keys())

    def get_event_log(self, topic: Optional[str] = None) -> List[Event]:
        """Returns the event log, optionally filtered by topic."""
        if topic:
            return [e for e in self._event_log if e.topic == topic]
        return list(self._event_log)

    def get_dead_letters(self) -> List[Event]:
        """Returns events that had no subscribers when published."""
        return list(self._dead_letter)

    def clear(self) -> None:
        """Resets all subscribers and logs."""
        self._subscribers.clear()
        self._event_log.clear()
        self._dead_letter.clear()
