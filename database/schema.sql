CREATE DATABASE IF NOT EXISTS retail_forecaster;
USE retail_forecaster;

CREATE TABLE products (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    item_id VARCHAR(50) NOT NULL,
    demo_name VARCHAR(100),
    category VARCHAR(50),
    department VARCHAR(50),
    store_id VARCHAR(20),
    state_id VARCHAR(10),
    UNIQUE KEY unique_item_store (item_id, store_id)
);

CREATE TABLE forecast_history (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    product_id INT,
    item_id VARCHAR(50),
    store_id VARCHAR(20),
    forecast_days INT,
    predicted_demand FLOAT,
    confidence FLOAT,
    revenue FLOAT,
    suggested_price FLOAT,
    current_inventory INT,
    safety_stock INT,
    days_until_stockout INT,
    reorder_quantity INT,
    ai_summary TEXT,
    forecast_data JSON,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE inventory (
    inventory_id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT,
    item_id VARCHAR(50),
    store_id VARCHAR(20),
    current_stock INT DEFAULT 100,
    safety_stock INT DEFAULT 20,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    UNIQUE KEY unique_inv_item_store (item_id, store_id)
);

CREATE TABLE product_aliases (
    alias_id INT AUTO_INCREMENT PRIMARY KEY,
    item_id VARCHAR(50) NOT NULL UNIQUE,
    demo_name VARCHAR(100) NOT NULL
);

INSERT INTO product_aliases (item_id, demo_name) VALUES
('item_1', 'Rice (Basmati)'),
('item_2', 'Milk (Fresh)'),
('item_3', 'Bread (Whole Wheat)'),
('item_4', 'Cooking Oil'),
('item_5', 'Sugar'),
('item_6', 'Salt'),
('item_7', 'Flour (All Purpose)'),
('item_8', 'Eggs'),
('item_9', 'Butter'),
('item_10', 'Tea Powder'),
('item_11', 'Coffee'),
('item_12', 'Juice'),
('item_13', 'Water Bottle'),
('item_14', 'Snacks (Chips)'),
('item_15', 'Soap'),
('item_16', 'Shampoo'),
('item_17', 'Toothpaste'),
('item_18', 'Detergent'),
('item_19', 'Cleaning Spray'),
('item_20', 'Paper Towels'),
('item_21', 'Tissues'),
('item_22', 'Trash Bags'),
('item_23', 'Candles'),
('item_24', 'Batteries'),
('item_25', 'Light Bulbs'),
('item_30', 'Biscuits'),
('item_35', 'Candy'),
('item_40', 'Cereal'),
('item_45', 'Pasta'),
('item_50', 'Canned Beans');
