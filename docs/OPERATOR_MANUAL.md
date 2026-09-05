# Clow Terminal: Operator & Trader Manual

## 1. Operating Modes

Clow supports three distinct operating modes configured via the Left Sidebar:

1. **Disabled (Default / Fail-Safe)**
   - All live and simulated order dispatching is strictly prohibited.
   - Ideal for analyzing charts, reviewing model predictions, and monitoring incoming MT5 tick feeds.

2. **Co-Pilot Mode (Recommended for Discretionary Traders)**
   - When Tactical AI Radar detects high conviction ($> 70\%$) and actionable R:R ($> 1.2$), an order proposal card is generated.
   - The card displays calculated lot size, exact cash risk (\$ and \%), entry, SL, and TP.
   - The trader reviews the card and clicks `[Approve]` to submit or `[Reject]` with an optional reason.

3. **Auto-Pilot Mode (Autonomous Systematic Execution)**
   - Automatically evaluates incoming AI signals every bar.
   - Passes candidate orders through the Sovereign Risk Gates (Drawdown, Max Open Positions, Spread Filter).
   - If approved, autonomously places pending limit/stop orders and manages expiration countdowns.

---

## 2. Emergency Panic Kill-Switch

- Located prominently on the Top Telemetry Header and Sidebar (`[PANIC KILL-SWITCH]`).
- Clicking the button or triggering programmatic risk breach executes:
  1. Instant cancellation of all pending orders across all pairs.
  2. Immediate market liquidation of all open positions.
  3. Sovereign lock on the risk manager, halting any new trade authorizations until manual human clearance.

---

## 3. Keyboard Shortcuts & Canvas Interactions

- **Pan / Zoom Chart:** Mouse wheel scroll to zoom price/time scale; Left-click drag to pan.
- **Crosshair Tool:** Middle-click to toggle crosshair inspect mode.
- **Modify Protection Levels:** Drag Stop Loss (SL) or Take Profit (TP) horizontal level lines directly on the chart canvas.
- **Close Position:** Single-click the `[Close]` button on any active position row in the Bottom Dock.
