#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Docker Compose Helper ===${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}File .env tidak ditemukan. Membuat dari .env.example...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}File .env berhasil dibuat. Silakan edit sesuai kebutuhan.${NC}"
        echo -e "${YELLOW}Jalankan script ini lagi setelah mengedit .env${NC}"
        exit 0
    else
        echo -e "${RED}File .env.example tidak ditemukan!${NC}"
        exit 1
    fi
fi

# Menu
echo ""
echo "Pilih aksi:"
echo "1) Start semua services"
echo "2) Start dengan rebuild"
echo "3) Stop semua services"
echo "4) View logs (semua)"
echo "5) View logs (webserver only)"
echo "6) Restart webserver"
echo "7) Down dan hapus volumes"
echo "8) Check status"
echo "0) Exit"
echo ""
read -p "Pilihan [0-8]: " choice

case $choice in
    1)
        echo -e "${GREEN}Starting services...${NC}"
        docker-compose up -d
        echo -e "${GREEN}Services started!${NC}"
        echo ""
        echo "Akses aplikasi di:"
        echo "  - Webserver: http://localhost:5000"
        echo "  - Grafana: http://localhost:3030"
        echo "  - InfluxDB: http://localhost:8086"
        ;;
    2)
        echo -e "${GREEN}Building and starting services...${NC}"
        docker-compose up -d --build
        echo -e "${GREEN}Services started!${NC}"
        ;;
    3)
        echo -e "${YELLOW}Stopping services...${NC}"
        docker-compose down
        echo -e "${GREEN}Services stopped!${NC}"
        ;;
    4)
        echo -e "${GREEN}Menampilkan logs (Ctrl+C untuk keluar)...${NC}"
        docker-compose logs -f
        ;;
    5)
        echo -e "${GREEN}Menampilkan logs webserver (Ctrl+C untuk keluar)...${NC}"
        docker-compose logs -f webserver
        ;;
    6)
        echo -e "${YELLOW}Restarting webserver...${NC}"
        docker-compose restart webserver
        echo -e "${GREEN}Webserver restarted!${NC}"
        ;;
    7)
        read -p "Yakin ingin hapus volumes? (y/n): " confirm
        if [ "$confirm" == "y" ]; then
            echo -e "${RED}Stopping and removing volumes...${NC}"
            docker-compose down -v
            echo -e "${GREEN}Done!${NC}"
        else
            echo -e "${YELLOW}Cancelled${NC}"
        fi
        ;;
    8)
        echo -e "${GREEN}Status services:${NC}"
        docker-compose ps
        ;;
    0)
        echo -e "${GREEN}Bye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Pilihan tidak valid!${NC}"
        exit 1
        ;;
esac
