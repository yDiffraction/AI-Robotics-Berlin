"""Send one sport request to the sim, the way action_bridge does: python sim/mujoco/send_action.py hello"""
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_api.msg.dds_ import (Request_, RequestHeader_, RequestIdentity_, RequestLease_,
                                                     RequestPolicy_, Response_)

from sport_actions import ACTIONS

SIM_DOMAIN_ID = 1

by_name = {name.split(" (")[0].replace(" ", "_"): api_id for api_id, (name, _) in ACTIONS.items()}
arg = sys.argv[1] if len(sys.argv) == 2 else ""
api_id = by_name.get(arg) or (int(arg) if arg.isdigit() else None)
if api_id is None:
    raise SystemExit(f"usage: send_action.py <api_id | {' | '.join(by_name)}>")

ChannelFactoryInitialize(SIM_DOMAIN_ID, "lo")
acks = []
sub = ChannelSubscriber("rt/api/sport/response", Response_)
sub.Init(lambda msg: acks.append((msg.header.identity.api_id, msg.header.status.code)), 10)
pub = ChannelPublisher("rt/api/sport/request", Request_)
pub.Init()
time.sleep(0.5)  # let discovery finish before the single write

# built by hand: unitree_sdk2py's default Request_ helper raises a TypeError
identity = RequestIdentity_(time.monotonic_ns(), api_id)
request = Request_(RequestHeader_(identity, RequestLease_(0), RequestPolicy_(0, False)), "", [])
pub.Write(request)
deadline = time.monotonic() + 2.0
while not acks and time.monotonic() < deadline:
    time.sleep(0.02)
print(f"api_id={api_id}: " + (f"ack code {acks[0][1]}" if acks else "no response - is sport_emulator running?"))
