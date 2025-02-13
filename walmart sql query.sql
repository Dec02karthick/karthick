SHOW databases;
USE walmart_db;
SHOW TABLES;
DESC walmart;
SHOW COLUMNS FROM walmart;
SELECT * FROM walmart LIMIT 10;
SELECT COUNT(*) FROM walmart;
select distinct payment_method from walmart;
select
      payment_method,
      count(*)
from walmart
group by payment_method

#DROP TABLES walmart;
select 
      count(distinct branch)
from walmart;

select max(quantity) from walmart;

# Business Problems

#Q1: Find different payment methods, number of transactions, and quantity sold by payment method
select
      payment_method,
      count(*) as no_payment,
      sum(quantity) as no_qty_sold
from walmart
group by payment_method

#Q2Identify the highest-rated category in each branch
-- Display the branch, category, and avg rating

select *
from (
     select
	    branch,
	    category,
	    avg(rating) as avg_rating,
	    rank() OVER (PARTITION BY branch ORDER BY AVG(rating) DESC) AS ranking
    from walmart
    group by branch, category
) AS ranked_data 
where ranking = 1;

#Q3 Identify the busiest day for each branch based on the number of transactions

select *
from
( select
    branch,
    date_format(str_to_date(date, '%d/%m/%y'), '%W') as day_name,
    count(*) as no_transactions,
    rank() over(partition by branch order by count(*) desc) as ranking
from walmart
group by branch, day_name
)AS ranked_data
where ranking =1;

#Q4 Calculate the total quantity of items sold per payment method

select
      payment_method,
      sum(quantity) as no_qty_sold
from walmart
group by payment_method

#Q5 Determine the average, minimum, and maximum rating of categories for each city.-
#list the city of avg, max,min rating.

select 
    city,
    category,
    min(rating) as min_rating,
    max(rating) as max_rating,
    avg(rating) as avg_rating
from walmart
group by city, category

#Q6 Calculate the total profit for each category

select
    category,
    sum(total) as total_revenue,
    sum(total * profit_margin) as profit
from walmart
group by category

#Q7  Determine the most common payment method for each branch

with cte as
(
  select
    branch,
    payment_method,
    count(*) as total_trans,
    rank() over (partition by branch order by count(*) desc) as ranking
 from walmart
 group by branch, payment_method
)
select *
from cte
where ranking =1;

#Q8  Categorize sales into Morning, Afternoon, and Evening shifts

select
    branch,
case 
      when hour (TIME(time)) < 12 then 'morning'
      when hour (TIME(time)) between 12 and 17 then 'afternoon'
      else 'evening'
  end as day_time,
  count(*)
from walmart
group by branch, day_time
order by branch, 3 desc

#Q9 Identify the 5 branches with the highest revenue decrease ratio from last year to current year (e.g., 2022 to 2023)

select *,
	year(str_To_date(date, '%d/%m/%y')) as year,
    str_To_date(date, '%d/%m/%y') as formated_date
from walmart;

with revenue_2022
as
(
  select
     branch,
     sum(total) as revenue
 from walmart
 where year(str_To_date(date, '%d/%m/%y')) = 2022
 group by branch
),
revenue_2023
as
(
  select
     branch,
     sum(total) as revenue
  from walmart
  where year(str_To_date(date, '%d/%m/%y')) = 2023
  group by branch
)
select
    ls.branch,
    ls.revenue as last_year_revenue,
    cs.revenue as cr_year_revenue,
    round(
         (cast(ls.revenue as decimal)- cast(cs.revenue as decimal)) /
         cast(ls.revenue as decimal) * 100, 2) as rev_dec_ratio
from revenue_2022 as ls
join
revenue_2023 as cs
on ls.branch = cs.branch
where 
     ls.revenue > cs.revenue
order by 4 desc
limit 5



