-- ================================================================
-- E-COMMERCE CUSTOMER BEHAVIOR & CHURN ANALYSIS
-- ================================================================
USE olist;

-- SECTION 1: DATA QUALITY AUDIT

-- 1a. Null check across all critical columns
SELECT COUNT(*) AS total_orders,
    SUM(CASE WHEN order_status IS NULL THEN 1 ELSE 0 END) AS null_status,
    SUM(CASE WHEN order_purchase_timestamp IS NULL THEN 1 ELSE 0 END) AS null_purchase_date,
    SUM(CASE WHEN order_delivered_customer_date IS NULL THEN 1 ELSE 0 END) AS null_delivery_date,
    ROUND(SUM(CASE WHEN order_delivered_customer_date IS NULL THEN 1 ELSE 0 END)* 100.0 / COUNT(*), 2) AS pct_missing_delivery
FROM olist_orders_dataset;

-- 1b. Order status breakdown
SELECT order_status,
    COUNT(*) AS order_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2)  AS pct_share
FROM olist_orders_dataset
GROUP BY order_status
ORDER BY order_count DESC;

-- 1c. Dataset date range
SELECT MIN(order_purchase_timestamp) AS earliest_order,
    MAX(order_purchase_timestamp) AS latest_order,
    DATEDIFF(MAX(order_purchase_timestamp),
             MIN(order_purchase_timestamp)) AS days_covered
FROM olist_orders_dataset;

-- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- SECTION 2: REVENUE ANALYSIS

-- 2a. Monthly revenue with MoM growth (Window Function: LAG)
WITH monthly_rev AS (
    SELECT DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS order_month,
        ROUND(SUM(p.payment_value), 2) AS revenue
    FROM olist_orders_dataset o
    JOIN olist_order_payments_dataset p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m')
)
SELECT
    order_month,
    revenue,
    LAG(revenue) OVER (ORDER BY order_month)  AS prev_month,
    ROUND((revenue - LAG(revenue) OVER (ORDER BY order_month)) * 100.0 /
        NULLIF(LAG(revenue) OVER (ORDER BY order_month), 0), 2) AS mom_growth_pct,
    ROUND(SUM(revenue) OVER (ORDER BY order_month ROWS UNBOUNDED PRECEDING), 2) AS cumulative_revenue
FROM monthly_rev
ORDER BY order_month;

-- 2b. Top 10 product categories by revenue
SELECT p.product_category_name,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(SUM(pay.payment_value), 2) AS total_revenue,
    ROUND(AVG(pay.payment_value), 2) AS avg_order_value,
    RANK() OVER (ORDER BY SUM(pay.payment_value) DESC) AS revenue_rank
FROM olist_orders_dataset o
JOIN olist_order_items_dataset oi ON o.order_id   = oi.order_id
JOIN olist_products_dataset p ON oi.product_id = p.product_id
JOIN olist_order_payments_dataset pay ON o.order_id = pay.order_id
WHERE o.order_status = 'delivered'
GROUP BY p.product_category_name
ORDER BY total_revenue DESC
LIMIT 10;

-- 2c. Revenue by payment type
SELECT p.payment_type,
    COUNT(DISTINCT o.order_id) AS orders,
    ROUND(SUM(p.payment_value), 2) AS total_revenue,
    ROUND(AVG(p.payment_value), 2) AS avg_order_value,
    ROUND(AVG(p.payment_installments), 1) AS avg_installments
FROM olist_orders_dataset o
JOIN olist_order_payments_dataset p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
GROUP BY p.payment_type
ORDER BY total_revenue DESC;

-- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- SECTION 3: CUSTOMER ANALYSIS

-- 3a. Customer purchase frequency — how many buy more than once?
SELECT order_count,
    COUNT(*) AS customers,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct_of_customers
FROM (
    SELECT c.customer_unique_id,
        COUNT(DISTINCT o.order_id) AS order_count
    FROM olist_orders_dataset o
    JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
) freq
GROUP BY order_count
ORDER BY order_count;

-- 3b. RFM Segmentation (full SQL implementation)
WITH snapshot AS (
    SELECT MAX(order_purchase_timestamp) + INTERVAL 1 DAY AS snap_date
    FROM olist_orders_dataset
    WHERE order_status = 'delivered'
),
rfm_raw AS (
    SELECT c.customer_unique_id,
        DATEDIFF((SELECT snap_date FROM snapshot),
                 MAX(o.order_purchase_timestamp)) AS recency,
        COUNT(DISTINCT o.order_id) AS frequency,
        ROUND(SUM(p.payment_value), 2) AS monetary
    FROM olist_orders_dataset o
    JOIN olist_customers_dataset c ON o.customer_id  = c.customer_id
    JOIN olist_order_payments_dataset p ON o.order_id  = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
),
rfm_scored AS (
    SELECT *,
        NTILE(4) OVER (ORDER BY recency  DESC) AS R,
        NTILE(4) OVER (ORDER BY frequency ASC) AS F,
        NTILE(4) OVER (ORDER BY monetary  ASC) AS M
    FROM rfm_raw
)
SELECT *,
    CASE
        WHEN R >= 4 AND F >= 3 AND M >= 3 THEN 'Champions'
        WHEN R >= 3 AND F >= 3            THEN 'Loyal Customers'
        WHEN R >= 4 AND F <= 1            THEN 'New Customers'
        WHEN R >= 3 AND F <= 2            THEN 'Potential Loyalists'
        WHEN R  = 2 AND F >= 2            THEN 'At Risk'
        WHEN R <= 2 AND F <= 2            THEN 'Churned'
        ELSE 'Needs Attention'
    END AS segment
FROM rfm_scored
ORDER BY monetary DESC;

-- 3c. Segment summary — average stats per group
WITH snapshot AS (
    SELECT MAX(order_purchase_timestamp) + INTERVAL 1 DAY AS snap_date
    FROM olist_orders_dataset WHERE order_status = 'delivered'
),
rfm_raw AS (
    SELECT
        c.customer_unique_id,
        DATEDIFF((SELECT snap_date FROM snapshot),
                 MAX(o.order_purchase_timestamp)) AS recency,
        COUNT(DISTINCT o.order_id) AS frequency,
        ROUND(SUM(p.payment_value), 2) AS monetary
    FROM olist_orders_dataset o
    JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
    JOIN olist_order_payments_dataset p ON o.order_id    = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
),
rfm_scored AS (
    SELECT *,
        NTILE(4) OVER (ORDER BY recency  DESC) AS R,
        NTILE(4) OVER (ORDER BY frequency ASC) AS F,
        NTILE(4) OVER (ORDER BY monetary  ASC) AS M
    FROM rfm_raw
),
rfm_segments AS (
    SELECT *,
        CASE
            WHEN R >= 4 AND F >= 3 AND M >= 3 THEN 'Champions'
            WHEN R >= 3 AND F >= 3            THEN 'Loyal Customers'
            WHEN R >= 4 AND F <= 1            THEN 'New Customers'
            WHEN R >= 3 AND F <= 2            THEN 'Potential Loyalists'
            WHEN R  = 2 AND F >= 2            THEN 'At Risk'
            WHEN R <= 2 AND F <= 2            THEN 'Churned'
            ELSE 'Needs Attention'
        END AS segment
    FROM rfm_scored
)
SELECT segment,
    COUNT(*)                        AS customers,
    ROUND(AVG(recency),   0)        AS avg_recency_days,
    ROUND(AVG(frequency), 1)        AS avg_orders,
    ROUND(AVG(monetary),  0)        AS avg_revenue_brl,
    ROUND(SUM(monetary),  0)        AS total_revenue_brl
FROM rfm_segments
GROUP BY segment
ORDER BY customers DESC;

-- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- SECTION 4: COHORT RETENTION

-- Monthly cohort retention table
WITH first_purchase AS (
    SELECT c.customer_unique_id,
        DATE_FORMAT(MIN(o.order_purchase_timestamp), '%Y-%m') AS cohort_month
    FROM olist_orders_dataset o
    JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
),
customer_activity AS (
    SELECT c.customer_unique_id,
        DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS order_month
    FROM olist_orders_dataset o
    JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
),
cohort_data AS (
    SELECT
        fp.cohort_month,
        PERIOD_DIFF(
            EXTRACT(YEAR_MONTH FROM STR_TO_DATE(CONCAT(ca.order_month,'-01'), '%Y-%m-%d')),
            EXTRACT(YEAR_MONTH FROM STR_TO_DATE(CONCAT(fp.cohort_month,'-01'), '%Y-%m-%d'))) AS period_number,
        COUNT(DISTINCT ca.customer_unique_id) AS customers
    FROM customer_activity ca
    JOIN first_purchase fp ON ca.customer_unique_id = fp.customer_unique_id
    GROUP BY fp.cohort_month, period_number
),
cohort_sizes AS (
    SELECT cohort_month, customers AS cohort_size
    FROM cohort_data
    WHERE period_number = 0
)
SELECT
    cd.cohort_month,
    cd.period_number,
    cd.customers,
    cs.cohort_size,
    ROUND(cd.customers * 100.0 / cs.cohort_size, 1) AS retention_pct
FROM cohort_data cd
JOIN cohort_sizes cs ON cd.cohort_month = cs.cohort_month
ORDER BY cd.cohort_month, cd.period_number;

-- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- SECTION 5: DELIVERY & SATISFACTION ANALYSIS

-- 5a. Avg delivery days by state + avg review score
SELECT c.customer_state,
    ROUND(AVG(DATEDIFF(o.order_delivered_customer_date,
                       o.order_purchase_timestamp)), 1)  AS avg_delivery_days,
    ROUND(AVG(r.review_score), 2) AS avg_review_score,
    COUNT(DISTINCT o.order_id) AS total_orders
FROM olist_orders_dataset o
JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
JOIN olist_order_reviews_dataset r ON o.order_id    = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
GROUP BY c.customer_state
ORDER BY avg_delivery_days DESC;

-- 5b. Late delivery impact on review scores
SELECT
    CASE
        WHEN DATEDIFF(order_delivered_customer_date,
                      order_estimated_delivery_date) <= 0  THEN 'On Time / Early'
        WHEN DATEDIFF(order_delivered_customer_date,
                      order_estimated_delivery_date) <= 7  THEN '1–7 Days Late'
        WHEN DATEDIFF(order_delivered_customer_date,
                      order_estimated_delivery_date) <= 14 THEN '8–14 Days Late'
        ELSE '15+ Days Late'
    END AS delivery_bucket,
    COUNT(*) AS orders,
    ROUND(AVG(r.review_score), 2) AS avg_review_score
FROM olist_orders_dataset o
JOIN olist_order_reviews_dataset r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY delivery_bucket
ORDER BY avg_review_score ASC;


-- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- SECTION 6: CHURN ANALYSIS IN SQL

-- 6a. Flag churned customers (no order in last 90 days)
WITH last_orders AS (
    SELECT c.customer_unique_id,
        MAX(o.order_purchase_timestamp) AS last_purchase,
        COUNT(DISTINCT o.order_id) AS total_orders,
        ROUND(SUM(p.payment_value), 2) AS lifetime_value
    FROM olist_orders_dataset o
    JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
    JOIN olist_order_payments_dataset p ON o.order_id    = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
),
snapshot AS (
    SELECT MAX(last_purchase) AS snap_date FROM last_orders
)
SELECT lo.*,
    DATEDIFF((SELECT snap_date FROM snapshot), lo.last_purchase) AS days_since_purchase,
    CASE
        WHEN DATEDIFF((SELECT snap_date FROM snapshot),
                      lo.last_purchase) > 90 THEN 'Churned'
        ELSE 'Active'
    END AS churn_status
FROM last_orders lo
ORDER BY days_since_purchase DESC;

-- 6b. Overall churn rate
WITH last_orders AS (
    SELECT c.customer_unique_id,
        MAX(o.order_purchase_timestamp) AS last_purchase
    FROM olist_orders_dataset o
    JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
),
snapshot AS (SELECT MAX(last_purchase) AS snap_date FROM last_orders)
SELECT
    SUM(CASE WHEN DATEDIFF((SELECT snap_date FROM snapshot),last_purchase) > 90 THEN 1 ELSE 0 END) AS churned,
    COUNT(*) AS total_customers,
    ROUND(SUM(CASE WHEN DATEDIFF((SELECT snap_date FROM snapshot),last_purchase) > 90 THEN 1 ELSE 0 END)* 100.0 / COUNT(*), 1) AS churn_rate_pct
FROM last_orders;

-- 6c. Churn rate by customer state
WITH last_orders AS (
    SELECT c.customer_unique_id,
        c.customer_state,
        MAX(o.order_purchase_timestamp) AS last_purchase
    FROM olist_orders_dataset o
    JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id, c.customer_state
),
snapshot AS (SELECT MAX(last_purchase) AS snap_date FROM last_orders)
SELECT
    customer_state,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN DATEDIFF((SELECT snap_date FROM snapshot),last_purchase) > 90 THEN 1 ELSE 0 END)  AS churned,
    ROUND(SUM(CASE WHEN DATEDIFF((SELECT snap_date FROM snapshot),last_purchase) > 90 THEN 1 ELSE 0 END)* 100.0 / COUNT(*), 1) AS churn_rate_pct
FROM last_orders
GROUP BY customer_state
ORDER BY churn_rate_pct DESC;