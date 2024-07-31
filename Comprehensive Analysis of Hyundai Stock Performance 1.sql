-- Comprehensive Analysis of Hyundai Stock Performance
/*
This collection of SQL queries represents a thorough examination of Hyundai's stock performance over time. 
The analysis covers various aspects of stock behavior, from basic statistical measures to advanced technical indicators. 
Here's an overview of what has been accomplished:

1. We begin with summary statistics to understand the overall price range and distribution.
2. Trading volume is analyzed on a monthly basis and we identify days with exceptionally high volumes.
3. Daily returns are calculated to measure short-term price movements.
4. Technical indicators such as moving averages (50-day and 200-day) and Volume Weighted Average Price (VWAP) are computed.
5. Volatility is assessed using weekly data to understand price fluctuations over time.
6. We explore the relationship between trading volume and price changes.
7. Price streaks and significant gaps between trading days are identified to spot potential trends and anomalies.
8. Yearly price ranges are calculated to understand long-term price behavior.
9. Bullish and bearish engulfing patterns are detected, which can be useful for technical trading strategies.
10. Lastly, we identify days with unusual trading volumes, which might indicate significant events affecting the stock.

This comprehensive set of analyses provides a multi-faceted view of Hyundai's stock performance, 
offering insights that could be valuable for investors, analysts, and researchers studying the company's 
financial dynamics in the stock market.
*/

# Overview of the Data
SELECT 
    MIN(close) AS min_close,
    MAX(close) AS max_close,
    AVG(close) AS avg_close,
    STD(close) AS std_close
FROM hyandai.hyandai_full;

# Monthly Average Volumes
SELECT 
    YEAR(date) AS year,
    MONTH(date) AS month,
    AVG(volume) AS avg_volume
FROM hyandai_full
GROUP BY YEAR(date), MONTH(date)
ORDER BY year, month;

# Top 10 Days with Highest Trading Volume
SELECT date, volume, close
FROM hyandai_full
ORDER BY volume DESC
LIMIT 10;

# Calculate Daily Returns 
 SELECT 
    date,
    close,
    LAG(close) OVER (ORDER BY date) AS prev_close,
    (close - LAG(close) OVER (ORDER BY date)) / LAG(close) OVER (ORDER BY date) * 100 AS daily_return
FROM hyandai_full;

# Moving Averages (50-day and 200-day)
SELECT 
    date,
    close,
    AVG(close) OVER (ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS MA_50,
    AVG(close) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS MA_200
FROM hyandai_full
ORDER BY date;

# Volatility (using weekly data)
SELECT 
    YEAR(date) AS year,
    STD(close) AS price_volatility
FROM hyanda_weekly
GROUP BY YEAR(date)
ORDER BY year;

# Correlation between Volume and Price Change (using monthly data)
WITH monthly_changes AS (
    SELECT 
        date,
        volume,
        (close - open) AS price_change
    FROM hyanda_monthly
)
SELECT 
    (COUNT(*) * SUM(volume * price_change) - SUM(volume) * SUM(price_change)) / 
    (SQRT((COUNT(*) * SUM(volume * volume) - SUM(volume) * SUM(volume)) * 
    (COUNT(*) * SUM(price_change * price_change) - SUM(price_change) * SUM(price_change)))) AS volume_price_correlation
FROM monthly_changes;


# Longest Streak of Consecutive Price Increases/Decreases
SET @prev_close = NULL;
SET @prev_direction = NULL;
SET @streak_id = 0;

WITH price_direction AS (
    SELECT 
        date,
        close,
        @direction := IF(@prev_close IS NULL, 0,
                         IF(close > @prev_close, 1,
                            IF(close < @prev_close, -1, 0))) AS direction,
        @streak_id := IF(@direction = @prev_direction, @streak_id, @streak_id + 1) AS streak_id,
        @prev_close := close,
        @prev_direction := @direction
    FROM hyandai_full
    ORDER BY date
),
streaks AS (
    SELECT 
        streak_id,
        MIN(date) AS streak_start,
        MAX(date) AS streak_end,
        COUNT(*) AS streak_length,
        IF(direction = 1, 'Increase', 'Decrease') AS trend
    FROM price_direction
    WHERE direction != 0
    GROUP BY streak_id, direction
)
SELECT 
    streak_start,
    streak_end,
    streak_length,
    trend
FROM streaks
ORDER BY streak_length DESC
LIMIT 5;

# Identifying Gaps (significant price jumps between consecutive trading days)
SET @prev_close = NULL;

SELECT 
    date,
    open,
    prev_close,
    gap_percent
FROM (
    SELECT 
        date,
        open,
        @prev_close AS prev_close,
        CASE 
            WHEN @prev_close IS NOT NULL 
            THEN (open - @prev_close) / @prev_close * 100 
            ELSE NULL 
        END AS gap_percent,
        @prev_close := close
    FROM hyandai_full
    ORDER BY date
) AS subquery
WHERE gap_percent IS NOT NULL AND ABS(gap_percent) > 2
ORDER BY ABS(gap_percent) DESC;

# Price Range Analysis
SELECT 
    YEAR(date) AS year,
    MIN(low) AS yearly_low,
    MAX(high) AS yearly_high,
    (MAX(high) - MIN(low)) / MIN(low) * 100 AS yearly_range_percent
FROM hyandai_full
GROUP BY YEAR(date)
ORDER BY year;

# Volume Weighted Average Price (VWAP)
SELECT 
    date,
    close,
    volume,
    SUM(close * volume) OVER (ORDER BY date) / SUM(volume) OVER (ORDER BY date) AS VWAP
FROM hyandai_full
ORDER BY date;

# Identify Bullish and Bearish Engulfing Patterns
SELECT 
    current_day.date,
    'Bullish Engulfing' AS pattern
FROM 
    hyandai_full current_day
JOIN 
    hyandai_full previous_day ON current_day.date = previous_day.date + INTERVAL 1 DAY
WHERE 
    current_day.close > current_day.open
    AND previous_day.close < previous_day.open
    AND current_day.close > previous_day.open
    AND current_day.open < previous_day.close

UNION ALL

SELECT 
    current_day.date,
    'Bearish Engulfing' AS pattern
FROM 
    hyandai_full current_day
JOIN 
    hyandai_full previous_day ON current_day.date = previous_day.date + INTERVAL 1 DAY
WHERE 
    current_day.close < current_day.open
    AND previous_day.close > previous_day.open
    AND current_day.close < previous_day.open
    AND current_day.open > previous_day.close

ORDER BY date;

# Identify Days with Unusual Volume
WITH volume_stats AS (
    SELECT 
        date,
        volume,
        AVG(volume) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS avg_volume,
        STDDEV(volume) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS std_volume
    FROM hyandai_full
)
SELECT 
    date,
    volume,
    avg_volume,
    (volume - avg_volume) / std_volume AS volume_z_score
FROM volume_stats
WHERE ABS((volume - avg_volume) / std_volume) > 2
ORDER BY ABS((volume - avg_volume) / std_volume) DESC;


