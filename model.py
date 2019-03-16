import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

reg_filepath = "RegularSeasonDetailedResults1.csv"
reg_data = pd.read_csv(reg_filepath)
turn_filepath = "NCAATourneyDetailedResults1.csv"
turn_data = pd.read_csv(turn_filepath)
seed_filepath = "NCAATourneySeeds1.csv"
seed_data = pd.read_csv(seed_filepath)

features = ["FG_attempts", "EFG", "FT_attempts", "FT_percent", "turnover_diff", "DReb_percent", "OReb_percent", "fouls", "seed", "outcome"] # Maybe add location, but an average would be weird. Seed definitely should be added.
overall_teamstat = pd.DataFrame(columns = features)

def get_teamstats(team_id, year):
	FG_attempts_list = []
	EFG_list = []
	FT_attempts_list = []
	FT_percent_list = []
	turnover_diff_list = []
	DReb_list = []
	OReb_list = []
	fouls_list = []
	#location_list = []
	for i in range(reg_data.shape[0]):
		if (reg_data.iloc[i, 2] == team_id) and (reg_data.iloc[i, 0] == year):
			FG_attempts = reg_data.iloc[i, 9]
			EFG = calculate_EFG(FG_attempts, reg_data.iloc[i, 8], reg_data.iloc[i, 10])
			FT_attempts = reg_data.iloc[i, 9]
			FT_attempts = reg_data.iloc[i, 13]
			FT_percent = reg_data.iloc[i, 12]/reg_data.iloc[i, 13]
			if (type(FT_percent) != int):
				FT_percent = 0
			turnover_diff = reg_data.iloc[i, 17] - reg_data.iloc[i, 30]
			DReb_percent = calculate_DReb_percent(reg_data.iloc[i, 15], reg_data.iloc[i, 27])
			OReb_percent = calculate_OReb_percent(reg_data.iloc[i, 14], reg_data.iloc[i, 28])
			fouls = reg_data.iloc[i, 20]
			#location = location_to_scalar(reg_data.iloc[i, 6], True)
		elif (reg_data.iloc[i, 4] == team_id) and (reg_data.iloc[i, 0] == year):
			FG_attempts = reg_data.iloc[i, 22]
			EFG = calculate_EFG(FG_attempts, reg_data.iloc[i, 21], reg_data.iloc[i, 23])
			FT_attempts = reg_data.iloc[i, 26]
			FT_percent = reg_data.iloc[i, 25]/reg_data.iloc[i, 26]
			turnover_diff = reg_data.iloc[i, 30] - reg_data.iloc[i, 17]
			DReb_percent = calculate_DReb_percent(reg_data.iloc[i, 28], reg_data.iloc[i, 14])
			OReb_percent = calculate_OReb_percent(reg_data.iloc[i, 27], reg_data.iloc[i, 15])
			fouls = reg_data.iloc[i, 33]
			#location = location_to_scalar(reg_data.iloc[i, 6], False)
		else:
			continue
		FG_attempts_list.append(FG_attempts)
		EFG_list.append(EFG)
		FT_attempts_list.append(FT_attempts)
		FT_percent_list.append(FT_percent)
		turnover_diff_list.append(turnover_diff)
		DReb_list.append(DReb_percent)
		OReb_list.append(OReb_percent)
		fouls_list.append(fouls)
	return [np.mean(FG_attempts_list), np.mean(EFG_list), np.mean(FT_attempts_list), np.mean(FT_percent_list), np.mean(turnover_diff_list), np.mean(DReb_list), np.mean(OReb_list), np.mean(fouls_list)]

def calculate_EFG(FG_attempts, twopoint, threepoint):
		if FG_attempts == 0:
			return 0
		return (twopoint + (1.5* threepoint))/FG_attempts

def calculate_DReb_percent(DReb, OReb_Oppos):
	return DReb/(DReb + OReb_Oppos)

def calculate_OReb_percent(OReb, DReb_Oppos):
	return OReb/(OReb + DReb_Oppos)

def location_to_scalar(location, winner):
	if winner:
		if location == "H":
			return 2
		if location == "N":
			return 1
		if location == "A":
			return 0
	if not winner:
		if location == "A":
			return 2
		if location == "N":
			return 1
		if location == "H":
			return 0

def get_seed(team_id, year):
	for i in range(seed_data.shape[0]):
		if (seed_data.iloc[i, 2] == team_id) and (seed_data.iloc[i, 0] == year):
			seed = seed_data.iloc[i, 1]
			break
	if seed[1] == "0":
		seednum = int(seed[2])
	else:
		seednum = int(seed[1] + seed[2])
	return seednum

def combine_stats(higher_stat, lower_stat):
	new_stats = []
	for i in range(len(higher_stat)):
		new_stats.append(higher_stat[i]-lower_stat[i])
	return new_stats

def make_game_dataframe():
	game_list = []
	for i in range(turn_data.shape[0]):
		year = turn_data.iloc[i, 0]
		high_team = max(turn_data.iloc[i, 2], turn_data.iloc[i, 4])
		low_team = min(turn_data.iloc[i, 2], turn_data.iloc[i, 4])
		high_team_stats = get_teamstats(high_team, year)
		low_team_stats = get_teamstats(low_team, year)
		match_stats = combine_stats(high_team_stats, low_team_stats)
		match_stats.append(get_seed(high_team, year)-get_seed(low_team, year))
		if high_team == turn_data.iloc[i, 2]:
			match_stats.append(1)
		else:
			match_stats.append(0)
		game_list.append(match_stats)
		print(i)
	game_df = pd.DataFrame(game_list,columns=features)
	game_df = game_df.dropna()
	return game_df

def main():
	game_df = make_game_dataframe()

	X = game_df[features[0:len(features)-2]]
	Y = game_df[features[len(features)-1]]
	accuracy_list = []
	for i in range(50):
		train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=i)
		model = GaussianNB()
		model.fit(train_X, train_Y)
		predictions = model.predict(test_X)
		for i in range(len(predictions)):
			predictions[i] = round(predictions[i],0)
		accuracy = accuracy_score(test_Y, predictions)
		accuracy_list.append(accuracy)
	print(np.mean(accuracy_list))

main()

'''
Needs to speed up when collecting game data. Currently at 8 seconds a step, let's get it down to 2.
'''

'''
Effective Field Goal Average = (Made 2pt + 1.5 * Made 3pt)/FGA
DReb = (Def Rebound/ (Def Rebound + Opponent's Off Rebound))
OReb = (Off Rebound/ (Off Rebound + Opponent's Def Rebound))
turnover_diff = our turnover - their turnover
'''
