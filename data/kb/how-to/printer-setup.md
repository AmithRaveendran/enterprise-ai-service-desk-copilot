---
id: KB-0006
title: Adding an Office Printer and Using Secure Print
category: how-to
service: Printing
tags: [printer, printing, secure-print, badge, office]
owner: Workplace Technology
last_updated: 2026-04-28
---

# Adding an Office Printer and Using Secure Print

All office printers use **Secure Print (FollowMe)**. You send jobs to a single print queue,
then release them at any printer by tapping your building badge. Jobs that are not released
are deleted after **24 hours**.

## Adding the printer on Windows

1. Make sure you are connected to the office network (**Halcyon-Corp** Wi-Fi or wired) or
   to the VPN.
2. Open **Settings > Bluetooth & devices > Printers & scanners**.
3. Click **Add device**. Select **HD-FollowMe** from the list.
4. If it is not listed, click **Add manually** and enter `\\print01.halcyon-dynamics.example\HD-FollowMe`.

## Adding the printer on macOS

1. Open **System Settings > Printers & Scanners > Add Printer**.
2. Select the **IP** tab and enter `print01.halcyon-dynamics.example`.
3. Protocol: **LPD**. Queue: `HD-FollowMe`. Use **Generic PostScript** as the driver.

## Releasing a job

1. Walk to any office printer and tap your badge on the card reader.
2. Select the jobs to print and press **Print**.
3. If your badge is not recognised, register it once by entering your username and password
   on the printer's touchscreen.

## Defaults and quotas

- Default is **double-sided, black and white**. Colour can be selected in print options.
- Colour printing is limited to **200 pages per month** per person unless your manager approves
  more.
- Guests cannot use Secure Print. Ask your host to print documents for them.

## Common problems

| Problem | Fix |
|---|---|
| Job does not appear at the printer | Check you are on the corporate network or VPN; jobs from Halcyon-Guest Wi-Fi are blocked |
| Printer shows "Paper jam" or "Toner low" | Report it on the printer's touchscreen using **Report Issue**, or raise a ticket under **Workplace > Printer Fault** |
| "Driver unavailable" on Windows | Remove the printer and add it again; drivers are installed automatically |
