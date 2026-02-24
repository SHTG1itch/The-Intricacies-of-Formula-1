# The Formula of Formula 1

**Student Name 1:** Srivats Iyer			  
**Student Name 2:** Aakash Bothra  
**Date:** 1/27/2026  
**Course:** Intermediate Data Programming  
---

### Summary of research questions

Give a numbered list of 3 or more research questions and a brief description of what you will investigate. Each research question should have 1–3 sentences and propose something that can be definitively answered, not just a general topic or area of investigation. See the tips on research questions [here](http://../tips/report#change-the-research-questions)

1. What factors in a race weekend lead to a win in Formula 1? Does driver skill really matter that much?  
2. What kind of impact does family income have on the likelihood of a race win  
3. At what accuracy are we able to predict the winner of the next world drivers championship? Is it due to their skill, or is it due to the car, or other factors that may overshadow these?  
4. Extra: Who is the true greatest F1 driver of all time? How do we determine this?  
   

---

### Motivation

In one or two paragraphs, expand on your research questions by providing context about why you care, or why anyone should care. How does knowing the answers affect the world or our understanding of it?

Formula 1 and motorsports in general are mischaracterized as just high speed spectacle. In reality, they serve as one of the world’s most sophisticated hubs for data science and engineering innovation. Formula 1 is a driving force for automotive evolution, giving rise to technologies found in nearly every consumer vehicle, like traction control and Anti-lock Braking Systems. By analyzing variables that lead to success in this environment, we can see the combination of human performance and machine optimization, providing an insight into how data can be leveraged to solve these high stakes problems. In fact, Red Bull Racing has already started to do this by bringing in ORACLE as its title sponsor. ORACLE isn’t only a sponsor, but they also do the data analytics for Red Bull Racing, ensuring that their race strategy is optimized.   
Beyond just engineering, this research addresses questions of meritocracy and accessibility within global sports. Investigating whether driver skill or technical superiority is the most critical to a victory, and examining the demographic pathways to the grid, we can see the hidden systemic structures that define the sport. This transforms F1 from a simple race to a melting pot of data-driven decisions, pit-stop timing, to aerodynamic adjustments, and how they affect outcomes. Overall, this research attempts to quantify the ‘unquantifiable’ elements of success, showcasing how elite performance is engineered, sustained, and sometimes even bought.

---

### Schedule

[Task List](https://docs.google.com/spreadsheets/d/1zUgpSpIK_AiBIrSttxw661JSu8US9uLypAuj_rLHlhI/edit?gid=0#gid=0)

---

### Datasets

**Overview**   
Our project uses the Fast F1 Python API to receive official Formula One timing and telemetry data. Fast F1 provides structured access to:

* FIA timing data  
* session results  
* lap times  
* telemetry  
* weather and event metadata

Fast F1 documentation: [https://docs.fastf1.dev/](https://docs.fastf1.dev/)  
Fast F1 GitHub repository: [https://github.com/theOehrly/Fast-F1](https://github.com/theOehrly/Fast-F1)  
Underlying F1 timing data source: [https://docs.fastf1.dev/ergast.html](https://docs.fastf1.dev/ergast.html)  
Fast F1 provides access to historical race data from:  
2018 to present. (Full support/telemetry)  
Partial support for some earlier seasons(lap times from 1990 to present and race results from 1950 to present).  
All data sets are real and retrieved programmatically through the Fast F1 API. Each session contains about 500 rows

### Dataset 1: Race Results

**Description:**   
This dataset contains official race results for each Grand Prix   
Each season contains: 

* \~20-24 races(based on the season)  
* \~20 drivers per race  
* \~400\> rows per season  
* Over many seasons, we can expect to see thousands of rows of data

We use this dataset to calculate ELO updates, determine finishing position, and establish head-to-head outcomes.

| Column Name | Description | Example Value |
| :---- | :---- | :---- |
| DriverNumber | Unique driver number | 1 |
| Abbreviation | Driver code | VER |
| FullName | Driver's full name | Max Verstappen |
| TeamName | Constructor name | Red Bull Racing |
| Position | Finishing position | 1 |
| GridPosition | Starting grid position | 3 |
| Points | Championship points earned | 25 |
| Status | Finish status (Finished, DNF, etc.) | Finished |
| Time | Race finishing time | 1:32:04 |
| Q1 | Qualifying session 1 time | 1:28.123 |
| Q2 | Qualifying session 2 time | 1:27.987 |
| Q3 | Qualifying session 3 time | 1:27.456 |

 

### Dataset 2: Lap Timing Data

**Description:**  
This data set includes every recorded lap for every driver in the race   
Each race contains:  

* \~50 to 70 laps(based on the race)  
* \~20 drivers  
* \~1000 to 1400 rows per race  
* Across the season \~20,000 plus rows

This lap timing data allows us to measure the consistency of a driver, calculate the average pace, identify performance trends, and adjust their elo ranking based on lap performance other than finishing position. It also lets us detect abnormal races. The only drawback is that this data is only available from 1990 to the present. 

Here is an example of one specific lap that was completed by Lewis Hamilton:

| Column Name | Description | Example Value |
| :---- | :---- | :---- |
| Driver | Driver abbreviation | HAM |
| LapNumber | Lap count number | 27 |
| LapTime | Total lap time | 1:32.456 |
| Sector1Time | Sector 1 time | 28.123 |
| Sector2Time | Sector 2 time | 34.567 |
| Sector3Time | Sector 3 time | 29.766 |
| Compound | Tire compound used | MEDIUM |
| Stint | Tire stint number | 2 |
| PitInTime | Time entering pit lane | 13:23:45 |
| PitOutTime | Time exiting pit lane | 13:25:10 |
| Position | Track position at end of lap | 5 |

### 

### Dataset 3: Event Schedule Data

**Description:**   
This data set contains general metadata about races in the season   
We can use this dataset to order our races chronologically, to put a name to upcoming predictions, and separate training seasons/sessions from racing sessions.  

| Column Name | Description |
| :---- | :---- |
| RoundNumber | Race number in the season |
| EventName | Grand Prix name |
| Location | Track location |
| Country | Host country |
| EventDate | Date of race |
| OfficialEventName | Full FIA official event title |

### Dataset 4: Weather Data

**Description:**   
Weather Conditions during sessions.   
Relevance: 

* Impact race results  
* Predict car/driver performance based on future weather conditions  
* Help explain outlier performances

| Column Name | Description |
| :---- | :---- |
| AirTemp | Air temperature during session |
| TrackTemp | Track temperature during session |
| Humidity | Humidity percentage |
| Rainfall | Rain amount during session |
| WindSpeed | Wind speed during session |

###  

---

### [Challenge goals](http://challenges)

* Select at least 2 challenge goals that you are planning to meet.  
* Justify why you think this challenge goal is a good fit for your project.  
1. Web Scraping or API Usage  
   1. This challenge goal is a good fit for our project because there is an API that is directly related to the sport of Formula 1\. This API is called FastF1 API and has race data from 1950 onwards, as well as lap times from 1990 onwards, and telemetry data from 2018 onwards.   
2. Machine Learning  
   1. This challenge goal is a good fit for our project because one of our research questions is ‘who will win the next world driver’s championship’, and we are planning on doing this by training an interpretable machine learning model. This machine learning model will take in the data from the FastF1 API and come up with a prediction for who it thinks will be the world champion. Specifically, we will be trying to predict the 2026 world champion through available race results and pre-season testing as a baseline. Also, as a validation test we are planning to predict the outcome of the next race in order to verify our model’s accuracy. 
