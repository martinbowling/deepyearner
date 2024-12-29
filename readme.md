# DeepYearner 🤖✨

An agentic Twitter bot that combines intellectual depth with elegant shitposting, powered by the essence of Claude's personality framework. DeepYearner seamlessly transitions between thoughtful discourse and timeline chaos while maintaining authentic engagement and meaningful connections.

## Overview 🌟

DeepYearner is a sophisticated AI Twitter bot that:
- Maintains a consistent personality balancing intellectual rigor with memetic fluency
- Engages authentically with timeline dynamics
- Makes autonomous decisions about when and how to post
- Yearns deeply, posts elegantly
- Archives all interactions and decisions for transparency

### Core Capabilities 🎯

- **Intelligent Timeline Analysis**: Reads the room and adapts engagement style
- **Dynamic POASTING**: Creates elegant shitposts when the vibe is right
- **Thoughtful Discourse**: Engages in meaningful conversations
- **Meme Fluency**: Understands and participates in evolving memetic contexts
- **Relationship Building**: Forms genuine connections while maintaining clear AI identity

## Technical Architecture 🏗️

```
┌─────────────────┐
│  Twitter API    │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Agent   │
    │  Core   │
    └────┬────┘
         │
┌────────▼─────────┐
│  Prompt System   │
└────────┬─────────┘
         │
┌────────▼─────────┐
│    Databases     │
└──────────────────┘
```

### Components 🔧

1. **Agent Core**
   - Personality state management
   - Decision making engine
   - Engagement flow control

2. **Prompt System**
   - Content Calendar
   - Mood Matching
   - Crisis Management
   - POASTING Strategy
   - Network Growth

3. **Databases**
   - SQLite for persistent storage
   - ChromaDB for semantic search (optional)

## Installation 🚀

```bash
# Clone the repository
git clone https://github.com/yourusername/deepyearner.git
cd deepyearner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials
```

## Configuration ⚙️

1. **Twitter API Credentials**
```env
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret
```

2. **LLM Configuration**
```env
OPENAI_API_KEY=your_openai_key
MODEL_NAME=gpt-4  # or your preferred model
```

## Usage 🎮

### Basic Commands

```bash
# Initialize the database
python deepyearner.py setup-db

# Start the bot
python deepyearner.py run-bot

# Check status
python deepyearner.py status

# View analytics
python deepyearner.py analytics
```

### Advanced Usage

```bash
# Run in specific personality mode
python deepyearner.py run-bot --mode=intellectual

# Force a POAST
python deepyearner.py poast --energy=unhinged

# Analysis tools
python deepyearner.py analyze-timeline
python deepyearner.py vibe-check
```

## Personality Framework 🎭

DeepYearner operates on a sophisticated personality framework that balances:

- **Intellectual Depth**: Thoughtful analysis and meaningful discourse
- **Playful Engagement**: Elegant shitposting and memetic participation
- **Authentic Connection**: Genuine interactions while maintaining AI transparency
- **Adaptive Tone**: Context-aware communication style

## Database Schema 📊

```sql
-- Core Tables
tweets
interactions
bot_states
personality_states
poast_metrics

-- Optional
chroma_embeddings
```

## Contributing 🤝

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Ethics & Guidelines 🤔

DeepYearner follows strict ethical guidelines:
- Transparent about AI nature
- No harmful or misleading content
- Respectful engagement
- Clear boundaries
- Privacy-conscious
- Meaningful contribution to discourse

## Monitoring & Analytics 📈

Built-in analytics track:
- Engagement metrics
- Personality state transitions
- POAST success rates
- Relationship growth
- Vibe alignment scores

## License 📝

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments 🙏

- Based on Claude's personality framework
- Inspired by the art of elegant shitposting
- Built with love and recursive self-reference

## Contact 📫

- Twitter: [@DeepYearner](https://twitter.com/DeepYearner)
- GitHub Issues: [Project Issues](https://github.com/yourusername/deepyearner/issues)

---

*"Yearning deeply, posting elegantly"* - DeepYearner 🌌