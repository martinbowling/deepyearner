# Version History

## 1.0.0 - Initial Release
- OAuth 2.0 support with proper token management
- Rate limit handling with header-based backoff
- User analysis system with auto-follow capability
- Batch user import from CSV
- Analysis based on up to 100 tweets per user
- Scoring system:
  - 40% engagement rate
  - 40% topic alignment
  - 20% interaction quality
- Auto-follow criteria:
  - Score ≥ 0.8 (auto-follow)
  - Score ≥ 0.6 with engagement rate ≥ 0.7 (auto-follow)

### Key Features
- Twitter API v2 support
- Rate limit compliance
- Automated user analysis
- Batch processing
- CSV import support
- List management
- Engagement tracking

### Technical Details
- Rate limits:
  - User tweets: 900/15min
  - Following: 50/15min
  - User lookup: 900/15min
- Database: SQLite
- Authentication: OAuth 2.0 with refresh token support 