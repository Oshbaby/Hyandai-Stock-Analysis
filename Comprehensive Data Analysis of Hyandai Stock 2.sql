-- Comprehensive Data Analysis of Hyandai Stock
/* This SQL script performs a comprehensive analysis of the Hyandai stock data, covering a wide range of descriptive statistics, date range exploration, daily price movement analysis, aggregation of weekly and monthly data, calculation of daily returns and volatility, identification of significant price movements, intraday price trend analysis, detection of trading day gaps, and examination of trading volume patterns.

Descriptive Statistics: The script starts by calculating various descriptive statistics for the dataset, including the total number of records, minimum and maximum values for different columns (Date, Open, High, Low, Close, Adjusted Close, and Volume).
Explore Date Range: It then extracts the earliest and latest dates in the dataset to understand the time period covered by the data.
Analyze Daily Price Movement: Next, the script calculates the daily price change and daily percentage change for each trading day, providing insights into the daily price fluctuations.
Aggregate Weekly and Monthly Data: The script aggregates the data to weekly and monthly levels, calculating various statistics such as minimum, maximum, and total volume for each week and month.
Daily Returns: The script calculates the daily returns based on the adjusted closing prices, which can be used for further financial analysis.
Daily Volatility: It also computes the daily volatility using the standard deviation of the adjusted closing prices over a rolling window of 5 days.
Identify Significant Price Movements: The script flags days with significant price movements, defined as a daily return greater than or equal to 2% in absolute value.
Analyze Intraday Price Trends: The script examines the intraday price trends, calculating the intraday volatility and intraday price change for each trading day.
Identify Gaps in Trading Days: The script detects gaps in trading days by comparing the current day's opening price with the previous day's closing price.
Analyze Trading Volume Patterns: Finally, the script analyzes the trading volume patterns, identifying days with above-average or below-average trading volume.
*/

# Descriptive Statistics:
SELECT 
    COUNT(*) AS total_records,
    MIN(Date) AS min_date,
    MAX(Date) AS max_date,
    MIN(Open) AS min_open,
    MAX(Open) AS max_open,
    MIN(High) AS min_high, 
    MAX(High) AS max_high,
    MIN(Low) AS min_low,
    MAX(Low) AS max_low,
    MIN(Close) AS min_close,
    MAX(Close) AS max_close,
    MIN(`Adj Close`) AS min_adj_close,
    MAX(`Adj Close`) AS max_adj_close,
    MIN(Volume) AS min_volume,
    MAX(Volume) AS max_volume
FROM hyandai_full;

# Explore Date Range
SELECT 
    MIN(Date) AS earliest_date,
    MAX(Date) AS latest_date
FROM hyandai_full;

# Analyze Daily Price Movement
SELECT 
    Date,
    Open,
    High,
    Low,
    Close,
    `Adj Close`,
    Volume,
    (Close - Open) AS daily_price_change,
    ((`Adj Close` - Open) / Open) * 100 AS daily_percent_change
FROM hyandai_full
ORDER BY Date ASC;

# Aggregate Weekly and Monthly Data
-- Weekly Data
SELECT 
    DATE_FORMAT(Date, '%Y-%W') AS week,
    MIN(Date) AS week_start_date,
    MAX(Date) AS week_end_date,
    MIN(Open) AS min_open,
    MAX(Open) AS max_open,
    MIN(High) AS min_high,
    MAX(High) AS max_high,
    MIN(Low) AS min_low,
    MAX(Low) AS max_low,
    MIN(Close) AS min_close,
    MAX(Close) AS max_close,
    MIN(`Adj Close`) AS min_adj_close,
    MAX(`Adj Close`) AS max_adj_close,
    SUM(Volume) AS total_volume
FROM hyandai_full
GROUP BY week
ORDER BY week_start_date ASC;

-- Monthly Data
SELECT
    DATE_FORMAT(Date, '%Y-%m') AS month,
    MIN(Date) AS month_start_date,
    MAX(Date) AS month_end_date,
    MIN(Open) AS min_open,
    MAX(Open) AS max_open,
    MIN(High) AS min_high,
    MAX(High) AS max_high,
    MIN(Low) AS min_low,
    MAX(Low) AS max_low,
    MIN(Close) AS min_close,
    MAX(Close) AS max_close,
    MIN(`Adj Close`) AS min_adj_close,
    MAX(`Adj Close`) AS max_adj_close,
    SUM(Volume) AS total_volume
FROM hyandai_full
GROUP BY month
ORDER BY month_start_date ASC;

# Daily Returns
SELECT 
    Date,
    Adj_Close,
    (Adj_Close - LAG(Adj_Close, 1) OVER (ORDER BY Date)) / LAG(Adj_Close, 1) OVER (ORDER BY Date) AS Daily_Return
FROM hyandai_full;

# Daily Volatility
SELECT
    Date,
    `Adj Close`,
    STDDEV(`Adj Close`) OVER (ORDER BY Date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS Daily_Volatility
FROM hyandai_full;

# Identify Significant Price Movements
SELECT
    Date,
    `Adj Close`,
    (`Adj Close` - LAG(`Adj Close`, 1) OVER (ORDER BY Date)) / LAG(`Adj Close`, 1) OVER (ORDER BY Date) AS Daily_Return,
    CASE
        WHEN ABS((`Adj Close` - LAG(`Adj Close`, 1) OVER (ORDER BY Date)) / LAG(`Adj Close`, 1) OVER (ORDER BY Date)) >= 0.02 THEN 'Significant Move'
        ELSE 'Normal Move'
    END AS Price_Movement_Flag
FROM hyandai_full;

# Analyze Intraday Price Trends
SELECT
    Date,
    Open,
    High,
    Low,
    Close,
    (High - Low) / Open * 100 AS Intraday_Volatility,
    (Close - Open) / Open * 100 AS Intraday_Price_Change
FROM hyandai_full;

# Identify Gaps in Trading Days
SELECT
    Date,
    Open,
    Close,
    LAG(Close, 1) OVER (ORDER BY Date) AS Prev_Close,
    Open - LAG(Close, 1) OVER (ORDER BY Date) AS Gap
FROM hyandai_full;

# Analyze Trading Volume Patterns
SELECT
    Date,
    `Volume`,
    `Adj Close`,
    (`Adj Close` - LAG(`Adj Close`, 1) OVER (ORDER BY Date)) / LAG(`Adj Close`, 1) OVER (ORDER BY Date) AS Daily_Return,
    CASE
        WHEN `Volume` > AVG(`Volume`) OVER (ORDER BY Date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) THEN 'Above Average'
        ELSE 'Below Average'
    END AS Volume_Status
FROM hyandai_full;