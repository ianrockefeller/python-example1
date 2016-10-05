import csv, time
# from operator import itemgetter, add
# from collections import Counter
import numpy as np
# import matplotlib.pyplot as plt
from get_survey_res import get_survey_res

"""
HOW THIS WORKS:
In Decipher Survey Builder,
Report > Data Downloads
Download in tab-delimited format

Set the file name (fname) to match *.tsv file
Make sure protocol_write_flag and output_write_flag are set to True
Include the Decipher survey.xml file
"""
# START USER MANUAL INPUT
fname = 'data.txt'
protocol_write_flag = True
output_write_flag = True

early_click = 4
middle_click = 4
late_click = 3
# END USER MANUAL INPUT

""""""""""""""""""""""""""""""
"""YOU SHALL NOT PASS!!!!!!"""
""""""""""""""""""""""""""""""
survey_info, output_col_names, levels = get_survey_res()

total_columns = survey_info['Total_Columns']
total_patient_cases = survey_info['Total_Patient_Cases']
attribute_totals = []
attribute_max_levels = []

for i in range(1, total_columns + 1):
	attribute_totals.append(survey_info['C%d_Total_Attributes' % i])

res_names = []

for i in range(1, total_columns + 1):
	at = attribute_totals[i-1]
	for j in range(1, at + 1):
		if output_col_names['C%d_Attribute_%d' % (i,j)]:
			res_names.append(output_col_names['C%d_Attribute_%d' % (i,j)])

def isint(n):
	try:
		int(n)
		return True
	except ValueError:
		return False

def isfloat(n):
	try:
		float(n)
		return True
	except ValueError:
		return False

def inv_table(table):
	return [list(t) for t in zip(*table)]

# get index of attribute/characteristic in 
# res list i.e. the order to be shown in output file
def get_res_index(attrs):
	idx = attrs.index(1)
	return idx % 12

def generate_med_headers():
	med_headers = []
	# create new headers for each cell in matrix
	for i in range(1, total_columns + 1):
		num_rows = attribute_totals[i-1]
		for j in range(1, num_rows + 1):
			med_headers.append('M%dA%d' % (i,j))

	return med_headers

def open_data(fname):
	data = {}

	with open(fname, 'r') as f:
		freader = csv.reader(f, delimiter='\t')
		headers = next(freader)

		for header in headers:
			data[header] = []
		
		for row in freader:	
			for r,header in zip(row,headers):
				data[header].append(r)

	return data

def get_survey_click_data(data, res_count, total_patient_cases, new_headers):
	survey_data = []
	choices_by_respondent = []
	
	for res in range(res_count): 						# per respondent
		res_data = []
		choices = []
		for pc in range(1, total_patient_cases + 1): 	# per loop iteration (patient case)
			pc_data = []
			last_step = 0
			for i in range(1, total_columns + 1): 	# per column
				num_rows = attribute_totals[i-1]
				for j in range(1, num_rows + 1): 		# per row 
					each_step = [] # [time, rank, levels]

					time = data['FINAL_medication%d_attribute%d_%dr1' % (i,j,pc)][res]
					rank = data['FINAL_medication%d_attribute%d_%dr2' % (i,j,pc)][res]

					if rank:
						rank = int(float(rank))

					each_step.append(data['record'][res])
					each_step.append(pc)
					each_step.append(rank) # append rank
					each_step.append(time) # append time

					each_step.append(data['Q_ISDT_SEARCH_%d_NT%dr%d' % (pc,i,j)][res]) # level

					# append all Attributes (all cells in matrix) to flag which was clicked
					for x in range(1, total_columns + 1):					
						for y in range(1, attribute_totals[x-1] + 1): # FIX THIS LINE!!!!!!!!!!!!!!!!!!!!!
							if x == i and y == j and time and rank:
								each_step.append(1)
							else:
								each_step.append(0)

					each_step.append('') # append treatment choice
					pc_data.append(each_step)

					# steps aren't in order, this gets the number of steps taken at a given PC
					if each_step[2]:
						curr_step = int(float(each_step[2]))
						if curr_step > last_step:
							last_step = int(float(each_step[2]))

			pc_data.sort(key=lambda x: x[new_headers.index('Step')]) # sort each step by rank to get click order
			choice = [' ' for i in range(len(new_headers))] # showing 1/0 for which was clicked for each cell in the matrix
			choice[0] = data['record'][res] # respondent ID
			choice[1] = pc # which loop iteration
			choice[2] = last_step + 1 # pseudo-step number for Choice
			choice[-1] = data['Q_ISDT_CHOICE_%dr1' % pc][res] # append choice to end, could do choice.append() as well
			choices.append(int(data['Q_ISDT_CHOICE_%dr1' % pc][res]))

			pc_data.append(choice)
			res_data.append(pc_data)
		choices_by_respondent.append(choices)
		survey_data.append(res_data)

	return [survey_data, choices_by_respondent]

def format_survey_data(survey_data, new_headers):
	output = []
	output.append(new_headers)
	for s_data in survey_data:
		for data in s_data:
			for row in data:
				if row[new_headers.index('Time')] and row[new_headers.index('Step')]: # if time and rank exists
					output.append(row)

	return output

def write_protocol_file(output):
	t = int(time.time())
	with open('iSDT_Protocol_Data_%d.csv' % t, 'wb') as csvfile:
	    writer = csv.writer(csvfile)
	    writer.writerows(output)

	return True

def write_output_file(output):
	# make this better!
	t = int(time.time())

	res_titles = ['Attributes'] + res_names # add Attribute column header

	table_builder = [res_titles] + output
	
	# build table to be shown
	output_table = [list(o) for o in zip(*table_builder)]

	with open('iSDT_Output_Data_%d.csv' % t, 'wb') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(output_table)

	return True

def get_clicks(output, new_headers):
	cols = inv_table(output) # flip rows to cols

	# remove first 5 columns and last column (Choice) 
	# to be left with just the click info per matrix cell
	cols = cols[len(new_headers):-1]

	clicks = [[float(c) for c in col if isint(c)] for col in cols] # trim some stuff

	return clicks

def get_click_totals(clicks):
	total_col_clicks = [reduce(lambda x,y: x+y, click) for click in clicks] # add the clicks of each col for col click totals 
	attr_cols = [] # attr_clicks broken up into lists by column
	first = 0
	last = attribute_totals[0]
	max_rows = 0

	for i in range(total_columns): # for each attribute column
		if attribute_totals[i] > max_rows:
			max_rows = attribute_totals[i]
		if i > 0:
			first = last 
			last = first + attribute_totals[i]
		attr_cols = attr_cols + (total_col_clicks[first:last]) # gets same index in each column

	return attr_cols

def get_share_total_clicks(click_totals):
	total_clicks = reduce(lambda x,y: x+y, click_totals) # get total click count
	percentages = [nc / total_clicks * 100 for nc in click_totals]

	return percentages

def format_share_total_clicks(percentages):
	return ['Share of Total Clicks'] + [str(p)+'%' for p in percentages]

def get_click_frequency(click_totals):
	total_pc = total_patient_cases * res_count

	click_frequency = map(lambda x: x / total_pc * 100, click_totals)

	return click_frequency

def format_click_frequency(frequency):
	return ['Click Frequency'] + [str(f)+'%' for f in frequency]

def get_avg_time(output, new_headers):
	# Alot of this is just data manipulation
	# cols is temporary to obtain times
	cols = inv_table(output) # flip rows to cols
	cols = cols[new_headers.index('Time')][1:] # leave out header

	times = [float(col) for col in cols if isfloat(col)] # get time column

	# clicks is temporary to obtain names
	clicks = get_clicks(output, new_headers)
	clicks = inv_table(clicks) # flip it back to rows

	name_idxs = [get_res_index(click) for click in clicks] # index of res_names for each row

	time_per_attrs = [[] for x in range(len(res_names))] # initialize

	for time,name_idx in zip(times, name_idxs): # organize time per attribute organized by index in res_names
		time_per_attrs[name_idx].append(time) 

	# Data manipulation done
	# Calculate the mean and std to check time values above a certain threshold

	ref = {}
	ref['mean'] = []
	ref['std'] = []

	# set mean and standard deviation
	for time_per_attr in time_per_attrs:
		ref['mean'].append(np.mean(time_per_attr))
		ref['std'].append(np.std(time_per_attr, ddof=1))

	# check if above threshold, if so replace with threshold value
	for mean,std,time_per_attr in zip(ref['mean'], ref['std'], time_per_attrs):
		threshold = mean + (2 * std)
		for time in time_per_attr:
			if time > threshold:
				xDim = time_per_attrs.index(time_per_attr) # makes testing easier
				yDim = time_per_attr.index(time) # makes testing easier
				time_per_attrs[xDim][yDim] = threshold

	# recalculate mean now within threshold
	mean = [np.mean(tpa) for tpa in time_per_attrs]

	return mean

def format_avg_time(data):
	return ['Average Time'] + [format(d, '.4f') for d in data]

def get_bucket_clicks(protocol, new_headers):
	protocol = protocol[1:] # trim off headers
	outputs = [] # formatted outputs for manipulation
	t_outputs = [] # temp

	# format data so each set of clicks are grouped per question
	for p in protocol:
		trimmed_p = p[len(new_headers):-1]
		if trimmed_p[0] == ' ':
			outputs.append(t_outputs)
			t_outputs = []
		else:
			t_outputs.append(trimmed_p)

	buckets = [[0,0,0] for x in range(reduce(lambda x,y: x+y, attribute_totals))]

	for output in outputs:
		rank = 0 # cell rank based on click order
		for o in output:
			idx = get_res_index(o)
			if rank > early_click:
				if rank > early_click + middle_click:
					bucket = 2
				else:
					bucket = 1
			else:
				bucket = 0
			buckets[idx][bucket] += 1
			rank += 1
	bucket_clicks = []
	for bucket in buckets:
		bucket_total = reduce(lambda x,y: x + y, bucket)
		bucket_click = []
		for b in bucket:
			bucket_click.append(float(b) / bucket_total)
		bucket_clicks.append(bucket_click)

	return bucket_clicks

def format_bucket_clicks(bucket_clicks):
	formatted_bucket_clicks = [[str(bc*100) + '%' for bc in bucket_click] for bucket_click in bucket_clicks]
	formatted_bucket_clicks.insert(0, ['Early', 'Middle', 'Late'])
	return inv_table(formatted_bucket_clicks)

def get_bw_data(data, res_count, total_patient_cases, total_columns, choices):
	bw_data = []

	# get best/worst data from decipher output
	for res in range(res_count): 						# per respondent
		res_data = {}
		for i in range(total_columns):
			res_data['treatment_%d_levels' % (i+1)] = []
		for pc in range(1, total_patient_cases + 1): 	# per loop iteration (patient case)
			for i in range(total_columns):
				i += 1
				if pc == 1:
					res_data['treatment_%d_best' % (i)] = []
					res_data['treatment_%d_worst' % (i)] = []
				res_data['treatment_%d_best' % (i)].append(data['Q_ISDT_BW_%d_%dc1' % (i, pc)][res])
				res_data['treatment_%d_worst' % (i)].append(data['Q_ISDT_BW_%d_%dc2' % (i, pc)][res])
				nt_levels = []
				for j in range(total_attributes):
					nt_levels.append(data['Q_ISDT_SEARCH_%d_NT%dr%d' % (pc, i, j+1)][res])
				res_data['treatment_%d_levels' % (i)].append(nt_levels)
		bw_data.append(res_data)
	
	bmw_tx_scores = []
	bmw_pt_scores = []
	# calculate Best/Worst scores
	for res in range(res_count):
		bmw_tx = [[0 for y in range(attribute_max_levels)] for x in range(total_attributes)]
		res_choices = choices[res] 
		res_tx_best_scores = [[] for i in range(total_attributes)]
		res_tx_worst_scores = [[] for i in range(total_attributes)]
			
		for pc in range(total_patient_cases): # len(res_choices) == total_patient_cases
			if res_choices[pc] <= total_columns: # a treatment was selected
				choice = int(res_choices[pc]) # treatment choice
				best = int(bw_data[res]['treatment_%d_best' % choice][pc]) # attribute as best
				worst = int(bw_data[res]['treatment_%d_worst' % choice][pc]) # attribute as worst
				blevel = int(bw_data[res]['treatment_%d_levels' % (choice)][pc][best-1]) # best level chosen
				wlevel = int(bw_data[res]['treatment_%d_levels' % (choice)][pc][worst-1]) # worst level chosen

				#best minus worst treatments
				bmw_tx[best-1][blevel-1] += 1 # best calculation
				bmw_tx[worst-1][wlevel-1] -= 1 # worst calculation
		for i in range(len(bmw_tx)):
			for j in range(len(bmw_tx[j])):
				bmw_tx[i][j] = bmw_tx[i][j] / float(total_columns) # normalize values

		tx_avg = [sum(row) / len(row) for row in bmw_tx] # averages per attribute per respondent
		bmw_tx_scores.append(tx_avg) # each row is a respondent
	
	raw_output = []

	bmw_tx_scores = inv_table(bmw_tx_scores) # make each row an attribute
	bw_tx_avgs = [sum(row) / len(row) for row in bmw_tx_scores]
	bmw_tx_avg_rescaled = [(avg - min(bw_tx_avgs)) / (max(bw_tx_avgs) - min(bw_tx_avgs)) for avg in bw_tx_avgs]

	raw_output += bmw_tx_avg_rescaled

	return raw_output

def format_bw(data):
	return ['B/W'] + [format(d, '.4f') for d in data]


# PROTOCOL
data = open_data(fname) # bring TSV file into Python

res_count = len(data['record']) # get number of respondents, i.e. how many records/rows in spreadsheet

new_headers = ['Respondent','Loop','Step', 'Time', 'Level'] # new output headers
med_headers = generate_med_headers() # start forming output table with column headers
full_headers = new_headers + med_headers + ['Choice']  # join lists

# organize data by column, key = column header
survey_data, choices = get_survey_click_data(data, res_count, total_patient_cases, full_headers) 

protocol = format_survey_data(survey_data, full_headers) # get subset of data you need

if protocol and protocol_write_flag:
	write_protocol_file(protocol) # write to csv file

# OUTPUT
output = []

# Total Clicks %
clicks = get_clicks(protocol, new_headers)

click_totals = get_click_totals(clicks) # bring characteristic clicks back in
raw_total_clicks = get_share_total_clicks(click_totals)
total_clicks = format_share_total_clicks(raw_total_clicks)

output.append(total_clicks)

# Click frequency
raw_click_frequency = get_click_frequency(click_totals)
click_frequency = format_click_frequency(raw_click_frequency)

output.append(click_frequency)

# Average time per attribute
raw_avg_time = get_avg_time(protocol, new_headers)
avg_time = format_avg_time(raw_avg_time)

output.append(avg_time)

# Bucketing clicks based on click order Early, Middle, Late
raw_bucket_clicks = get_bucket_clicks(protocol, new_headers)
bucket_clicks = format_bucket_clicks(raw_bucket_clicks)

for bucket_click in bucket_clicks:
	output.append(bucket_click)
"""
# B/W scores
raw_bw_data = get_bw_data(data, res_count, total_patient_cases, total_columns, choices)
bw_data = format_bw(raw_bw_data)

output.append(bw_data)
"""

# OUTPUT THE OUTPUT
if output and output_write_flag:
	write_output_file(output)

print 'Done!'

"""
TODO:
change all output to str, some are int

# bar graph example/test
y_pos = np.arange(len(res_names))
performance = 3 + 10 * np.random.rand(len(res_names))
error = np.random.rand(len(res_names))

plt.barh(y_pos, performance, align='center', alpha=0.4)
plt.yticks(y_pos, res_names)
plt.xlabel('Percent')
plt.title('Share of Total Clicks')

plt.show()
"""