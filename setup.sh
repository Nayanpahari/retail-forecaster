#!/bin/bash

echo "========================================="
echo " AI Retail Demand Forecaster - Setup"
echo "========================================="

echo ""
echo "Step 1: Setting up database..."
if command -v mysql &> /dev/null; then
    mysql -u root -p < database/schema.sql 2>/dev/null || echo "Please set up MySQL database manually"
else
    echo "MySQL not found. Please install MySQL and run: mysql -u root -p < database/schema.sql"
fi

echo ""
echo "Step 2: Setting up backend..."
cd backend
pip install -r requirements.txt
cd ..

echo ""
echo "Step 3: Setting up frontend..."
cd frontend
npm install
cd ..

echo ""
echo "Step 4: Setting up ML pipeline..."
cd ml
pip install -r requirements.txt
cd ..

echo ""
echo "========================================="
echo " Setup Complete!"
echo "========================================="
echo ""
echo "To start the application:"
echo "  Backend:  cd backend && uvicorn app.main:app --reload"
echo "  Frontend: cd frontend && npm run dev"
echo "  Docker:   cd docker && docker-compose up --build"
echo ""
