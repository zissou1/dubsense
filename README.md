# Dubsense

**Dubsense** is a lightweight monitoring tool with a user-friendly GUI that continuously checks for victory notifications in *Call of Duty: Warzone*. When a victory is detected, the tool can send a webhook notification to a specified endpoint, making it ideal for players and teams who want real-time alerts.

## Features

- **Graphical User Interface**: Easy-to-use interface for managing monitoring settings and configuring webhooks.
- **Continuous Monitoring**: The tool runs in the background, constantly monitoring for victory events.
- **Webhook Support**: Configurable webhook allows sending notifications to various platforms, such as Discord, Slack, or custom endpoints.
- **Lightweight**: Designed to be fast and resource-efficient, making it suitable for prolonged use.
- **Easy Setup**: Simple configuration through the GUI.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/dubsense.git
    cd dubsense
    ```

2. **Install Dependencies**:  
    This project requires Python 3.10.2 and some additional libraries. Install them using:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Tool**:
    ```bash
    python dubsense.py

4. **Build**:
    ```bash
    python build_exe_pyinstaller.py
    ```
   If building fails, check build_exe_pyinstaller.py and check that the path to mklml is ok.

## Usage

1. Launch Dubsense by running the command above, or build/download .exe.
2. Configure the webhook URL and other settings directly in the GUI.
3. Start monitoring, and Dubsense will notify you through the specified webhook when a victory is detected.

## Configuration

All settings can be adjusted through the GUI.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

