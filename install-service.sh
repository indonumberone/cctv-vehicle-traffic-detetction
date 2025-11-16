#!/bin/bash
# Install script untuk CCTV Vehicle Detection Service

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CCTV Vehicle Detection Service Installer${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

# Get the directory where script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVICE_NAME="vehicle-detection"
SERVICE_FILE="${SCRIPT_DIR}/${SERVICE_NAME}.service"

echo -e "${YELLOW}[1/5]${NC} Checking service file..."
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${RED}Error: Service file not found: $SERVICE_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Service file found"

echo -e "${YELLOW}[2/5]${NC} Creating service user..."
if ! id "pengmas-yolo" &>/dev/null; then
    echo "Creating user 'pengmas-yolo'..."
    useradd -r -s /bin/false -c "CCTV Vehicle Detection Service" pengmas-yolo
    echo -e "${GREEN}✓${NC} User 'pengmas-yolo' created"
else
    echo -e "${GREEN}✓${NC} User 'pengmas-yolo' already exists"
fi

echo -e "${YELLOW}[3/5]${NC} Checking Python virtual environment..."
VENV_PATH="${SCRIPT_DIR}/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not found at $VENV_PATH${NC}"
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✓${NC} Virtual environment created"
    
    echo "Installing dependencies..."
    "$VENV_PATH/bin/pip" install --upgrade pip
    "$VENV_PATH/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${GREEN}✓${NC} Virtual environment found"
    
    # Check if dependencies are installed
    if ! "$VENV_PATH/bin/python" -c "import cv2, ultralytics, supervision" 2>/dev/null; then
        echo -e "${YELLOW}Warning: Some Python dependencies are missing${NC}"
        echo "Installing dependencies..."
        "$VENV_PATH/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"
        echo -e "${GREEN}✓${NC} Dependencies installed"
    else
        echo -e "${GREEN}✓${NC} All dependencies are installed"
    fi
fi

echo -e "${YELLOW}[4/5]${NC} Installing systemd service..."
cp "$SERVICE_FILE" /etc/systemd/system/
systemctl daemon-reload
echo -e "${GREEN}✓${NC} Service installed"

echo -e "${YELLOW}[5/5]${NC} Setting up permissions..."
# Allow pengmas-yolo user to access workspace and output directories
chown -R pengmas-yolo:pengmas-yolo "${SCRIPT_DIR}/output" 2>/dev/null || true
chmod -R 755 "${SCRIPT_DIR}/output" 2>/dev/null || true
# Allow read access to workspace files
chmod -R o+r "${SCRIPT_DIR}" 2>/dev/null || true
echo -e "${GREEN}✓${NC} Permissions configured"

echo -e "${YELLOW}[6/6]${NC} Enabling service to start on boot..."
systemctl enable ${SERVICE_NAME}.service
echo -e "${GREEN}✓${NC} Service enabled"

echo -e "${YELLOW}Info:${NC} Service status..."
systemctl status ${SERVICE_NAME}.service --no-pager || true

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Service commands:"
echo -e "  ${YELLOW}Start service:${NC}   sudo systemctl start ${SERVICE_NAME}"
echo -e "  ${YELLOW}Stop service:${NC}    sudo systemctl stop ${SERVICE_NAME}"
echo -e "  ${YELLOW}Restart service:${NC} sudo systemctl restart ${SERVICE_NAME}"
echo -e "  ${YELLOW}View status:${NC}     sudo systemctl status ${SERVICE_NAME}"
echo -e "  ${YELLOW}View logs:${NC}       sudo journalctl -u ${SERVICE_NAME} -f"
echo -e "  ${YELLOW}Disable service:${NC} sudo systemctl disable ${SERVICE_NAME}"
echo ""
echo -e "${YELLOW}Note:${NC} Service will automatically restart on failure"
echo ""
