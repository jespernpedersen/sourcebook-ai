
# Sourcebook AI

**An AI-powered Discord bot for retrieving and quoting answers from sourcebooks.**

  

![GitHub](https://img.shields.io/badge/Language-Python-blue)

![GitHub](https://img.shields.io/badge/Framework-Discord.py-green)

![GitHub](https://img.shields.io/badge/AI-DeepSeek%20R1%20Distill%20Qwen%201.5B-orange)


## Overview

Sourcebook AI is a **Discord bot** that uses **AI** to retrieve and provide answers to user questions based on a provided sourcebook (e.g., a markdown file). It sources its answers directly from the content of the sourcebook and includes references for transparency. Whether you're looking up rules for a game, technical documentation, or general knowledge, Sourcebook AI makes it easy to find accurate, sourced answers.

## Features

-  **AI-Powered Answers**: Uses advanced natural language processing (NLP) to understand and respond to user questions.

-  **Semantic Search**: Retrieves the most relevant section of the sourcebook using semantic search.

-  **Quote-Based Responses**: Provides answers with direct quotes and sources (e.g., section headers).

-  **Easy to Use**: Accessible to all users in a Discord channel via simple commands.

-  **Customizable Sourcebook**: Works with any markdown file, making it adaptable to different use cases.

## How It Works

1. A user asks a question in Discord using the `/ask` command.

2. The bot uses **semantic search** to find the most relevant section of the sourcebook.

3. It uses the **DeepSeek-R1-Distill-Qwen-1.5B** model to generate a concise and factual answer based on the retrieved context.

4. The bot sends the answer back to the user, along with the source section header for reference.

## Example Usage

### User Input

    /ask What is a campaign?

### Bot Response
**Answer:** A campaign is a series of interconnected adventures that form a larger story, typically led by a Dungeon Master and played by a group of characters. These adventures can span multiple sessions and often involve a mix of combat, exploration, and role-playing.
**Source:** Campaigns, Player's Handbook

## Setup Instructions  
### Prerequisites  
- Python 3.10  
- A Discord bot token (obtained from the [Discord Developer Portal](https://discord.com/developers/applications))  
- A markdown file containing the sourcebook content (e.g., `phb_sourcebook.md`)  

### Installation  
1. Clone the repository:  
   ```
   git clone https://github.com/your-username/sourcebook-ai.git
   cd sourcebook-ai
   ```
2.  Install the required dependencies:
    ```pip install -r requirements.txt```
    
3.  Create a  `.env`  file in the root directory and add your Discord bot token:
    
    ``DISCORD_TOKEN=your-discord-bot-token``
    
4.  Place your sourcebook markdown file (e.g.,  `phb_sourcebook.md`) in the project directory.
    
5.  Run the bot:
    
    ``python bot.py``

    
## Configuration

### Customizing the Sourcebook

-   Replace the  `phb_sourcebook.md`  file with your own markdown file.
    
-   Ensure the file is structured with clear section headers (e.g.,  `## Section Name`) for optimal semantic search results.
    

### Changing the AI Model

-   To use a different model, update the  `model_name`  variable in the  `bot.py`  file. For example:
    
    ``model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"``

----------

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
    
2.  Create a new branch for your feature or bug fix.
    
3.  Submit a pull request with a detailed description of your changes.
    

----------

## License

This project is licensed under the MIT License. See the  [LICENSE](https://license/)  file for details.

----------

## Acknowledgments

-   [Hugging Face](https://huggingface.co/)  for providing the DeepSeek-R1-Distill-Qwen-1.5B model.
    
-   [Discord.py](https://discordpy.readthedocs.io/)  for the Discord bot framework.
    
-   [Sentence-Transformers](https://www.sbert.net/)  for semantic search capabilities.
