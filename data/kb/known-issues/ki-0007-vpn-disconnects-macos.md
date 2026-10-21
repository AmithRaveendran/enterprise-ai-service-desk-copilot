---
id: KI-0007
title: "Known Issue: SecureLink VPN Disconnects Every Few Minutes on macOS 15"
category: known-issue
service: VPN
status: open
tags: [vpn, securelink, macos, disconnect, known-issue]
owner: Network Operations
last_updated: 2026-10-08
---

# Known Issue: SecureLink VPN Disconnects Every Few Minutes on macOS 15

**Status:** Open – workaround available. Vendor fix expected in SecureLink 6.4.

## Symptoms

- On Macs running macOS 15.x with SecureLink 6.2 or 6.3, the VPN connects successfully but
  drops after 3 to 10 minutes.
- The SecureLink log shows `tunnel keepalive timeout`.
- Reconnecting works, but the drop repeats.
- Windows laptops are **not** affected.

## Cause

A change in macOS 15 power management puts the network extension to sleep when the laptop is
on battery power and the screen is locked, which interrupts the VPN keepalive.

## Workaround

1. Open **System Settings > Battery > Options**.
2. Turn **on** "Prevent automatic sleeping on power adapter when the display is off".
3. Turn **off** "Enable Power Nap".
4. Open SecureLink, go to **Settings > Connection**, and change the protocol from
   **Auto** to **TCP**. This uses the fallback gateway `vpn-tcp.halcyon-dynamics.example`.
5. Disconnect and reconnect the VPN.

Most users report that the TCP setting alone stops the disconnects.

## Should I raise a ticket?

You do not need to raise a ticket if the workaround works. If the drops continue after the
workaround, raise a ticket under **Network > VPN**, mention **KI-0007**, and attach the
SecureLink log (**Help > Export Logs**). Tickets linked to this known issue are handled as P3.
