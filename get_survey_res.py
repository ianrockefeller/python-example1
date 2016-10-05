from bs4 import BeautifulSoup as Soup
import re

def get_survey_res():
	soup = Soup(open('survey.xml'), "lxml-xml")

	labels_to_find = ['Total_Patient_Cases']

	res_names = []
	output_col_names = {}
	survey_info = {}

	res_tags = soup.find_all('res')
	survey_info['Total_Columns'] = int(soup.find_all('res', {'label': 'Total_Columns'})[0].text)

	for i in range(survey_info['Total_Columns']):
		labels_to_find.append('C%d_Total_Attributes' % (i+1))
		labels_to_find.append('C%d_Attribute_Max_Levels' % (i+1))

	for i in range(survey_info['Total_Columns']):
		for res_tag in res_tags:
			label = res_tag['label']

			if label in labels_to_find:
				if res_tag.text:
					survey_info[labels_to_find[labels_to_find.index(label)]] = int(res_tag.text)

			attr_search = re.search('C%d_Attribute_\d+' % (i+1), label)
			if attr_search:
				if res_tag.text:
					output_col_names[str(attr_search.string)] = str(res_tag.text)

	levels = {}

	for x in range(survey_info['Total_Columns']):
		for i in range(survey_info['C%d_Total_Attributes' % (x+1)]):
			level = 0
			for j in range(survey_info['C%d_Attribute_Max_Levels' % (x+1)]):
				res = soup.find_all('res', {"label" : "C%d_A%d_L%d" % (x+1,i+1, j+1)}) # should only be one match
				if res:
					if res[0].text:
						level += 1
			levels['C%d_Attribute_%d' % (x+1,i+1)] = level			

	return [survey_info, output_col_names, levels]
