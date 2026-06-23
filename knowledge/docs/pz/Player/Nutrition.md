---
title: "Nutrition"
category: Player
type: Nutrition
source_url: https://pzwiki.net/wiki/Nutrition
scraped_at: 2026-02-24 16:23:41
method: archived_wiki_markdown
---

# Nutrition

Player Moodle • Skill • Trait • Occupation • Inventory • Nutrition • Sleep • Condition • Health • Knox Infection

Nutrition is a game mechanic in Project Zomboid redesigned and implemented in Build 34 . Each item of food has a quality known as nutritional value which is defined by four variables: carbohydrates , proteins , fats , and calories . Each of these variables currently only has an impact on the player's weight, which can affect fitness experience gain, endurance, speed, and fragility, while proteins give the “Protein Boost” for exercise . However, there were previous mentions that it would change the player in various other ways, such as strength and mental state , which are not currently implemented.

## Occupations & traits

### Occupations

### Fitness Instructor Fitness Instructor
- **Name:** Fitness Instructor Fitness Instructor
- **Starting points:** -6
- **Major skills:** +2 Running  +3 Fitness
- **Description:** Starts with the Nutritionist trait

### Traits

### Nutritionist
- **Name:** Nutritionist
- **Cost:** -4
- **Starting weight:** N/A
- **Description:** Can see the nutritional values of any food
### Very High Weight
- **Name:** Very High Weight
- **Cost:** +10
- **Starting weight:** 105
- **Description:** Slower running speed. Tire from running more easily, -2 Fitness
### High Weight
- **Name:** High Weight
- **Cost:** +6
- **Starting weight:** 95
- **Description:** Slower running speed. Tire from running more easily, -1 Fitness
### Low Weight
- **Name:** Low Weight
- **Cost:** +6
- **Starting weight:** 70
- **Description:** Low strength, low endurance, and prone to injury
### Very Low Weight
- **Name:** Very Low Weight
- **Cost:** +10
- **Starting weight:** 60
- **Description:** Very low strength, very low endurance, and prone to injury
### Emaciated
- **Name:** Emaciated
- **Cost:** N/A
- **Starting weight:** 50
- **Description:** Low strength, low endurance, and prone to injury  Not available during character creation

The Nutritionist: Can see the nutritional values of any food. nutritionist trait is only available during character creation , costing 4 trait points, or can be picked up for free when choosing the fitness instructor occupation . The nutritionist trait allows the player to "see the nutritional values of any food", which can normally only be seen on food that is packaged .

## Nutritional value

Nutritional value defines how healthy a particular food is and the effects they'll have on the player's weight. By consuming just one type of food, the player may find themselves suffering from weight loss or weight gain, therefore altering their effectiveness to perform certain actions.

### Weight

Weight is a player statistic presented on the Info panel, which can be toggled with the J key by default. During character creation, the player can choose between one of four traits, which will determine their starting weight; if none are chosen, they will begin with the default weight of 80. Over time the player's weight may begin to vary from its starting value, which depends on the nutritional value of the food they have been consuming and the level of exercise they've been doing. As the player's weight fluctuates, their traits will also change depending on their current weight. The game recognizes that there are actually five different traits relating to weight, rather than just the four that can be picked up during character creation, whereas the default/normal weight is considered the absence of a trait .

The different effects caused on the player, along with their weight ranges, can be seen in the table below.

### Very High Weight Very High Weight
- **Name:** Very High Weight Very High Weight
- **Weight range:** 100 or more
### High Weight High Weight
- **Name:** High Weight High Weight
- **Weight range:** 85 - 99.9
### Normal
- **Name:** Normal
- **Weight range:** 75.1 - 84.9
### Low Weight Low Weight
- **Name:** Low Weight Low Weight
- **Weight range:** 65.1 - 75
### Very Low Weight Very Low Weight
- **Name:** Very Low Weight Very Low Weight
- **Weight range:** 50.1 - 65
### Emaciated Emaciated
- **Name:** Emaciated Emaciated
- **Weight range:** 50 or less
**Name:** Degeneration Degeneration Degeneration 
 (Damage) | **Weight range:** 35

Note that the emaciated trait and degeneration currently only exist in the game mechanics and therefore have no visual representation in-game. 
 Also note that the game rounds values ​​to whole numbers, e.g., 99.7 will be displayed as 100.

Having any of the five traits listed above (excluding normal ) gives the player an attribute recognized as “has weight trouble.” A player with this attribute will not gain any fitness experience beyond level 6, whereas having the emaciated, very high weight, or very low weight trait, they will lose the ability to gain fitness experience altogether, regardless of the current level. Experience will return to normal upon losing/gaining some weight, i.e., removing the trait.

The game files ( Nutrition.class ) contain a base weight of 60 for females, with varying values for gaining and losing weight; however, this has not been implemented.

#### Gaining weight

Gaining weight is usually undesirable, as it grants the player slower movement speed and loss of fitness. However, weight gain may be needed if the player is already underweight, which has its set of negative effects. While exercise doesn't directly affect weight gain, it does contribute to burning calories and thus reducing weight. Therefore, gaining weight can be assisted by walking instead of running or sprinting, not climbing (either over fences or through windows), and sleeping often. Sleeping will expend the least amount of calories compared to any other activity. For this reason, taking the restless sleeper or sleepyhead traits is recommended for players that often have trouble with losing too much weight.

##### Food consumption for weight gain

The foods consumed by the player are the main contributor to weight gain. Calories must be above the minimumThreshold , which will scale with the player's current weight, thus making it more difficult to gain weight the higher their weight is. The minimumThreshold can be calculated with the following equation.

baseCalorieThreshold + ((currentWeight − 80) × 40)

**baseCalorieThreshold:** 700 with slow metabolism and if weight < 90.

The amount of weight gained is affected by the amount of calories, carbohydrates, and fats consumed along with the time that has passed. This can be presented as the following equation.

△Weight = weightGainFactor × calorieProportion × timeElapsed

**weightGainFactor:** The carbohydrates and fats that the player consumes will accumulate and decrease slowly over time. If the player's consumed carbohydrates or fats are 400 or less, weightGainFactor will be 0.000013, but between 400 and 700, it'll become 0.000026, and above 700, will be 0.000039.

**calorieProportion:** Like carbohydrates and fats, calories will accumulate and decrease slowly over time, which is directly affected by the amount of exercise and type of exercise performed. calorieProportion is calculated by dividing the currentCalories by 4000. 4000 is the maximum number of calories. Therefore, if the currentCalories is above 4000, the calorieProportion will be 1.

**timeElapsed:** This value is the amount of in-game seconds that have passed. Therefore, 1 hour would be 3600.

**Example:** A player with a weight of 85 maintains 3000 calories, 500 carbohydrates and 200 fats over 1 day.

It should be noted that these values are after calculating for calories burned from exercise, and are not taken directly from the foods consumed, nor is it realistic that these values will remain static over a day. These values are constantly changing, and thus the weight gained is also updating constantly.

#### Losing weight

Too much weight loss can lead to reduced strength and endurance, and the player can become more prone to injury. Losing weight is desirable if the player happens to be of high weight. Weight can be lowered by climbing over fences and through windows, chopping trees, sprinting instead of walking, and avoiding sleep, which can be helped with the wakeful trait. The biggest factor in losing weight is the type of food eaten; avoiding food would lead to death. The best foods to eat are those low in calories and high in hunger, such as the following fruits and vegetables: radishes , tomatoes , broccoli , carrots , onions , strawberries , berries , and cherries . It will often take the player a while before they begin to lose weight after dieting. This is due to any excess calories that were consumed previously being used first. In addition to gaining underweight traits from losing too much weight, if the player reaches a weight of 35, they will begin taking damage until some weight is gained. This is usually very rare. However, it can be caused by eating only fruits/vegetables for long periods of time.

The weight loss occurs if currentCalories are less than calorieDeficitThreshold .

**calorieDeficitThreshold:** This is calculated as (currentWeight − 70) × 30 . This value is capped at 0.

△Weight_Loss = 0.0000085 × calorieDeficitProportion × timeElapsed

**calorieDeficitProportion:** This is calculated as currentCalories / 2500. This proportion is capped at 1.

### Nutrients

**Carbohydrates:** Carbohydrates are used in determining potential weight gain. This value is ignored unless the player has consumed more than 400 or 700 carbohydrates, resulting in a multiplier of 2 or 3, respectively. If consumed carbohydrates <400, it still may be a multiplier of 2 or 3 based on fats.

**Proteins:** Protein surplus or malus affects the amount of strength experience gained by various tasks such as exercise.

(As of Build 34.5, Protein values between 50 and 300 provide a 1.5 multiplier to Str XP gain. Likewise, values below -300 apply a .7 gain penalty.)

**Fats:** Fats, previously called lipids, are used in determining potential weight gain. This value is ignored unless the player has consumed more than 400 or 700 fats, resulting in a multiplier of 2 or 3, respectively. If consumed fats <400, it still may be a multiplier of 2 or 3 based on carbohydrates.

**Calories:** Calories are the main weight-determining nutrient. This value must be balanced with hunger consistently, or else obesity or starvation may sneak up on the player.

## See also

- Agriculture
- Cooking
- Evolved recipes
- Fishing
- Food
- Health
- Trapping

## References
