Disclaimer:
This tool is for educational and analytical purposes only.
Not financial advice.

## Deribit Inverse Option Calculator (BTC & ETH)

A live option pricing and scenario analysis tool for Deribit inverse options,
using forward-based Black-Scholes (Black-76) pricing.

### Features
- BTC & ETH inverse options
- Forward-based pricing (matches Deribit methodology)
- ATM strike auto-selection
- Live IV, forward, and mark price comparison
- Scenario analysis with custom forward & IV

### Tech Stack
- Python
- Streamlit
- Deribit Public API
- SciPy

### Notes
- Uses Deribit `underlying_price` as forward (F)
- No API keys required (public endpoints only)
