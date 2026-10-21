---
id: KI-0012
title: "Known Issue: No Audio in Teams Calls When Using a Docking Station"
category: known-issue
service: Teams
status: open
tags: [teams, audio, headset, docking-station, known-issue]
owner: Collaboration Services
last_updated: 2026-09-29
---

# Known Issue: No Audio in Teams Calls When Using a Docking Station

**Status:** Open – workaround available. Firmware update for the docking stations is being
tested and will be deployed automatically.

## Symptoms

- When a ThinkPad is connected to a **Lenovo USB-C Dock Gen 2**, other participants in Teams
  calls cannot hear you, or you cannot hear them.
- Audio works when the laptop is undocked.
- The problem started after the September Windows update.

## Cause

After the update, Windows sometimes selects the dock's audio output as the default device
instead of the headset. Teams follows the Windows default.

## Workaround

1. In a Teams call, click **More (…) > Settings > Device settings**.
2. Under **Speaker** and **Microphone**, select your headset explicitly instead of
   "Default device".
3. To make it permanent, open **Settings > System > Sound** in Windows and set your headset as
   the default input and output device.
4. If the headset still is not listed, unplug the dock, wait 10 seconds and plug it back in.

## Should I raise a ticket?

No ticket is needed if the workaround works. If audio still fails, raise a ticket under
**Collaboration > Teams Issue** and mention **KI-0012** together with the dock's serial number
(printed on the underside).
