#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
  @ Time     : 19-4-13 上午10:59
  @ Author   : Vodka
  @ File     : Features.py
  @ Software : PyCharm
"""
import sys
import json
import torch
import re
import numpy as np
from Transform2vec import Transform2vec
from PredictDomain import Rnn
from gensim.models import KeyedVectors
reload(sys)
sys.setdefaultencoding('utf8')


class Features:
    def __init__(self):
        # For domain
        self.domain_types = ['Legislature', 'GrandPrix', 'Athlete', 'Person', 'Settlement', 'PopulatedPlace',
                             'SpaceMission', 'School', 'Species', 'Place', 'Station', 'FormulaOneRacer', 'Planet',
                             'PokerPlayer', 'WrittenWork', 'Department', 'Canal', 'Disease', 'Reference', 'SkiResort',
                             'Comedian', 'LaunchPad', 'River', 'Film', 'Spacecraft', 'Island', 'GolfPlayer', 'Region',
                             'Organisation', 'Broadcaster', 'File', 'LegalCase', 'YearInSpaceflight', 'SportsEvent',
                             'Bridge', 'Aircraft', 'Food', 'ChristianDoctrine', 'MilitaryPerson', 'TelevisionEpisode',
                             'Event', 'Olympics', 'MilitaryUnit', 'Regency', 'Boxer', 'PoliticalParty', 'Play',
                             'Airline', 'ChartsPlacements', 'MusicGenre', 'ReligiousBuilding', 'AnatomicalStructure',
                             'TennisPlayer', 'Airport', 'Mountain', 'ArchitecturalStructure', 'GridironFootballPlayer',
                             'Road', 'Work', 'MeanOfTransportation', 'Company', 'Memorial', 'Museum', 'Sales',
                             'Country', 'MilitaryConflict', 'Artist', 'Album', 'Saint', 'AutomobileEngine',
                             'PowerStation', 'Actor', 'SoccerPlayer', 'RouteOfTransportation', 'RomaniaSettlement',
                             'Grape', 'Mill', 'AdministrativeRegion', 'CollegeCoach', 'Project', 'ProtectedArea',
                             'Family', 'Openswarm', 'RadioStation', 'CelestialBody', 'IceHockeyPlayer', 'Agent',
                             'Statistic', 'FormerMunicipality', 'Municipality', 'TelevisionShow', 'Hotel',
                             'FictionalCharacter', 'Automobile', 'NorwaySettlement', 'Scientist', 'FigureSkater',
                             'Animal', 'Infrastructure', 'Cemetery', 'HistoricPlace', 'SnookerPlayer', 'Race',
                             'Athlete,_CareerStation', 'Instrument', 'Language', 'Galaxy', 'Constellation',
                             'SoccerLeagueSeason', 'BaseballPlayer', 'Ship', 'Opera', 'Letter', 'Brain',
                             'ChemicalElement', 'StatedResolution', 'PersonFunction', 'Weapon', 'ConcentrationCamp',
                             'RecordLabel', 'Cartoon', 'WineRegion', 'Protein', 'Shrine', 'UndergroundJournal',
                             'SpaceShuttle', 'Hospital', 'University', 'AcademicJournal', 'Beverage', 'Building',
                             'Parish,_Deanery', 'MemberResistanceMovement', 'SportsTeam', 'ResearchProject',
                             'Architect', 'Currency', 'SportCompetitionResult', 'EducationalInstitution', 'Blazon',
                             'Software', 'PeriodicalLiterature', 'PublicTransitSystem', 'LawFirm', 'Rocket',
                             'ChemicalSubstance', 'SoccerClub', 'Activity', 'Biomolecule', 'RaceHorse', 'Single',
                             'Song', 'Writer', 'Artwork', 'ChessPlayer', 'College', 'Globularswarm', 'Cricketer',
                             'OlympicResult', 'GermanSettlement', 'Document', 'RestArea', 'Band', 'SoccerTournament',
                             'Magazine', 'SportsLeague', 'Criminal', 'MusicalWork', 'Library', 'HungarySettlement',
                             'GeneLocation', 'Cleric', 'TermOfOffice', 'Train', 'Colour', 'BelgiumSettlement', 'Sport',
                             'Stream', 'FilmFestival', 'Nerve', 'Case', 'PenaltyShootOut', 'Royalty', 'NobleFamily',
                             'LiechtensteinSettlement', 'Skyscraper', 'Newspaper', 'FormulaOneRacing',
                             'ArchitecturalStructure,_Monument', 'EthnicGroup', 'Monument', 'WrestlingEvent',
                             'SiteOfSpecialScientificInterest', 'MultiVolumePublication', 'MythologicalFigure', 'Flag',
                             'RouteStop', 'Restaurant', 'BodyOfWater', 'SubMunicipality', 'Drug', 'RaceTrack',
                             'Locomotive', 'CareerStation', 'Theatre', 'Monastry', 'Plant',
                             'MilitaryConflict_,_NaturalEvent', 'VolleyballPlayer', 'AdultActor',
                             'Organisation,_Parish', 'WorldHeritageSite', 'City', 'Wrestler', 'OfficeHolder',
                             'Diocese,_Parish', 'Monarch', 'MilitaryConflict,_AdministrativeRegion', 'ChemicalCompound',
                             'Organisation,_PopulatedPlace', 'MusicalArtist', 'TelevisionStation', 'Law',
                             'MeanOfTransportation_,_Instrument', 'Astronaut', 'Continent', 'Lake', 'Volcano',
                             'SwitzerlandSettlement', 'Swimmer', 'WaterwayTunnel', 'Mayor',
                             'SupremeCourtOfTheUnitedStatesCase', 'Coach', 'LebanonSettlement', 'SportsTeamMember',
                             'NuclearPowerStation', 'Bishop', 'District', 'Election', 'VideoGame', 'Mountain,Volcano',
                             'Painting', 'Intercommunality', 'Artery', 'Gene', 'GraveMonument', 'CyclingTeam',
                             'Musical', 'Priest', 'Muscle', 'RoadJunction', 'Cave', 'Politician',
                             'NationalCollegiateAthleticAssociationAthlete', 'GivenName',
                             'ClericalAdministrativeRegion', 'CityDistrict', 'SpaceStation', 'HistoricBuilding',
                             'RailwayTunnel', 'GolfCourse', 'On-SiteTransportation', 'Engine', 'Escalator',
                             'ConveyorSystem', 'MovingWalkway', 'GeopoliticalOrganisation', 'LunarCrater'] # 269
        self.domain_dic = {}
        self.tt = []

        # For type
        self.class_vec = ''
        self.property_vec = ''

        # For range
        self.range_types = ['PoliticalParty', 'nonNegativeInteger', 'integer', 'string', 'PersonFunction',
                           'SpaceShuttle', 'positiveInteger', 'gYear', 'date', 'double', 'Person',
                           'PeriodicalLiterature', 'Group', 'Place', 'Award', 'Altitude', 'Rocket', 'Country',
                           'PopulatedPlace', 'Agent', 'Species', 'langString', 'SpaceMission', 'File', 'RaceTrack',
                           'GrossDomesticProductPerCapita', 'Airport', 'MeanOfTransportation', 'MusicGenre',
                           'Broadcaster', 'EducationalInstitution', 'Organisation', 'boolean', 'MountainRange', 'Work',
                           'Road', 'Vein', 'SportsTeam', 'Canal', 'Artist', 'City', 'anyURI', 'float',
                           'BroadcastNetwork', 'Plant', 'BoxingStyle', 'WineRegion', 'Cemetery', 'Lymph', 'HockeyTeam',
                           'TermOfOffice', 'Stadium', 'SpaceStation', 'Genre', 'List', 'Animal', 'Infrastructure',
                           'Actor', 'Company', 'TelevisionShow', 'MusicalArtist', 'GrandPrix', 'Language',
                           'Agglomeration', 'AnatomicalStructure', 'Organ', 'MusicalWork', 'MilitaryConflict',
                           'RecordLabel', 'Athlete', 'Ideology', 'Deity', 'BodyOfWater', 'Island', 'Event', 'Diocese',
                           'ChemicalSubstance', 'SocietalEvent', 'Sea', 'RadioStation', 'MilitaryUnit', 'Building',
                           'RouteOfTransportation', 'dateTime', 'Project', 'Automobile', 'Concept', 'Mountain',
                           'TimePeriod', 'gYearMonth', 'Settlement', 'RouteStop', 'Demographics', 'Population', 'River',
                           'SportsLeague', 'Jockey', 'Area', 'fuelType', 'Sound', 'College', 'FillingStation', 'Artery',
                           'GovernmentType', 'SoccerClub', 'EthnicGroup', 'Film', 'TeamMember', 'SpatialThing',
                           'Spacecraft', 'Municipality', 'Locomotive', 'CareerStation', 'SoccerTournament',
                           'Tournament', 'Diploma', 'Judge', 'Thing', 'Family', 'FormerMunicipality', 'Band',
                           'Province', 'Athletics', 'Currency', 'Magazine', 'BoxingCategory', 'Comic', 'WrittenWork',
                           'Contest', 'Continent', 'RaceHorse', 'Image', 'Museum', 'OlympicEvent', 'Depth', 'Arena',
                           'Saint', 'ClericalOrder', 'CultivatedVariety', 'Drama', 'OlympicResult', 'Sport',
                           'SportCompetitionResult', 'Astronaut', 'Deanery', 'Monarch', 'Sales', 'Legislature',
                           'School', 'Constellation', 'Race', 'Embryology', 'Colour', 'Taxon', 'SystemOfLaw',
                           'TrainCarriage', 'SportsEvent', 'engineConfiguration', 'Pope', 'Newspaper', 'Album', 'Nerve',
                           'Community', 'LaunchPad', 'Gene', 'OrganisationMember', 'Mayor', 'valvetrain', 'Novel',
                           'GeneLocation', 'Caterer', 'SoccerLeagueSeason', 'SoccerPlayer', 'HorseRiding', 'Architect',
                           'OrderedCollection', 'Grape', 'Instrument', 'Church', 'Annotation', 'AutomobileEngine',
                           'millimetre', 'kilogram', 'squareKilometre', 'cubicKilometre', 'cubicMetre',
                           'gramPerKilometre', 'kilometre', 'day', 'metre', 'hour', 'cubicMetrePerSecond', 'kelvin',
                           'minute', 'inhabitantsPerSquareKilometre', 'newtonMetre', 'kilogramPerCubicMetre', 'litre',
                           'kilometrePerHour', 'megabyte', 'cubicCentimetre', 'second', 'kilometrePerSecond',
                           'kilowatt', 'centimetre', 'squareMetre'] # 206
        self.range_dic = {}
        self.rt = []

        self.trans = Transform2vec()
        self.processData()

    def getDomainVec(self, uri):
        """
        Return the vector of domain for given uri
        :param uri: 
        :return: 
        """
        if uri in self.domain_dic.keys():
            domain_words = self.domain_dic[uri]
            pattern = "[A-Z]"
            clean_words = []
            new_string = re.sub(pattern, lambda x: " " + x.group(0), domain_words)
            ss = new_string.split(' ')
            for _s in ss:
                if _s != ' ' and _s != '':
                    _s = _s.strip('-')
                    _s = _s.strip('_')
                    clean_words.append(_s)
            _ = ' '.join(clean_words)
            # print _
            domain = self.trans.transform2onevec(_)
        else:
            model = Rnn(50, 50, 2, 269)
            model = torch.load('model/domain_predictor.model')
            words_in_key = uri.split('/')
            ww = words_in_key[-1]
            words_in_key = ww.split('#')
            clean_words = []
            for item in words_in_key:
                if item == "http:" or item == 'dbpedia.org' or item == '' or item == 'www.w3.org':
                    continue
                else:
                    pattern = "[A-Z]"
                    new_string = re.sub(pattern, lambda x: " " + x.group(0), item)
                    ss = new_string.split(' ')
                    for _s in ss:
                        if _s != ' ' and _s != '':
                            _s = _s.strip('-')
                            _s = _s.strip('_')
                            clean_words.append(_s)
            _ = ' '.join(clean_words)
            res_vec = self.trans.transform2onevec(_)
            res_vec = res_vec.reshape(1, 1, 50)
            res_vec = torch.from_numpy(res_vec)
            pred = model(res_vec)
            _, out = torch.max(pred, 1)  # get the index
            domain_words = self.domain_types[out.numpy()[0]]
            pattern = "[A-Z]"
            clean_words = []
            new_string = re.sub(pattern, lambda x: " " + x.group(0), domain_words)
            ss = new_string.split(' ')
            for _s in ss:
                if _s != ' ' and _s != '':
                    _s = _s.strip('-')
                    _s = _s.strip('_')
                    clean_words.append(_s)
            _ = ' '.join(clean_words)
            # print _
            domain = self.trans.transform2onevec(_)
        return domain

    def getRangeVec(self, uri):
        """
        Return the vector of range for given uri
        :param uri: 
        :return: 
        """
        if uri in self.range_dic.keys():
            range_words = self.range_dic[uri]
            pattern = "[A-Z]"
            clean_words = []
            new_string = re.sub(pattern, lambda x: " " + x.group(0), range_words)
            ss = new_string.split(' ')
            for _s in ss:
                if _s != ' ' and _s != '':
                    _s = _s.strip('-')
                    _s = _s.strip('_')
                    clean_words.append(_s)
            _ = ' '.join(clean_words)
            # print _
            range = self.trans.transform2onevec(_)
        else:
            model = Rnn(50, 50, 2, 269)
            model = torch.load('model/range_predictor.model')
            words_in_key = uri.split('/')
            ww = words_in_key[-1]
            words_in_key = ww.split('#')
            clean_words = []
            for item in words_in_key:
                if item == "http:" or item == 'dbpedia.org' or item == '' or item == 'www.w3.org':
                    continue
                else:
                    pattern = "[A-Z]"
                    new_string = re.sub(pattern, lambda x: " " + x.group(0), item)
                    ss = new_string.split(' ')
                    for _s in ss:
                        if _s != ' ' and _s != '':
                            _s = _s.strip('-')
                            _s = _s.strip('_')
                            clean_words.append(_s)
            _ = ' '.join(clean_words)
            res_vec = self.trans.transform2onevec(_)
            res_vec = res_vec.reshape(1, 1, 50)
            res_vec = torch.from_numpy(res_vec)
            pred = model(res_vec)
            _, out = torch.max(pred, 1)  # get the index
            range_words = self.range_types[out.numpy()[0]]
            pattern = "[A-Z]"
            clean_words = []
            new_string = re.sub(pattern, lambda x: " " + x.group(0), range_words)
            ss = new_string.split(' ')
            for _s in ss:
                if _s != ' ' and _s != '':
                    _s = _s.strip('-')
                    _s = _s.strip('_')
                    clean_words.append(_s)
            _ = ' '.join(clean_words)
            # print _
            range = self.trans.transform2onevec(_)
        return range

    def getTypeVec(self, uri):
        """
        Return the vector of type for given uri
        :param uri: 
        :return: 
        """
        names = uri.split('/')
        name = names[-1]
        names = name.split('#')
        name = names[-1]
        if len(name)>0 and name[0] >= 'A' and name[0] <= 'Z':
            return self.class_vec
        else:
            return self.property_vec

    def getUriVec(self, uri):
        """
        Return the vector of uri(only the last one phrase) for given uri
        :param uri: 
        :return: 
        """
        names = uri.split('/')
        name = names[-1]
        names = name.split('#')
        name = names[-1]
        clean_words = []
        pattern = "[A-Z]"
        new_string = re.sub(pattern, lambda x: " " + x.group(0), name)
        ss = new_string.split(' ')
        for _s in ss:
            if _s != ' ' and _s != '':
                _s = _s.strip('-')
                _s = _s.strip('_')
                clean_words.append(_s)
        _ = ' '.join(clean_words)
        # print _
        res_vec = self.trans.transform2onevec(_)
        return res_vec

    def extractInfoFromLcquad(self):
        """
        Extract questions from lc-quad
        :return: 
        """
        f = open('./data/lcquad.json')
        info = json.load(f)
        f.close()
        q = {}
        for i, item in enumerate(info):
            q[i] = item['question']
        with open('./data/questions.json', 'w') as json_file:
            json.dump(q, json_file)
        print q

    def processData(self):
        """
        Create some basic datas for Feature instance
        :return: 
        """

        # original data that have domain feature
        f = open('./data/clean_domain.txt')
        for line in f.readlines():
            line = line.strip('\n')
            d = line.split(' ')
            d[0] = d[0].strip('\r')
            d[1] = d[1].strip('\r')
            self.domain_dic[d[0]] = d[1]
            if d[1] not in self.tt:
                self.tt.append(d[1])
        f.close

        # subPropertyOf data to generate domain feature from parent property
        f = open('./data/clean_subPropertyOf.txt')
        for line in f.readlines():
            line = line.strip('\n')
            d = line.split(' ')
            d[0] = d[0].strip('\r')
            d[1] = d[1].strip('\r')
            if (d[0] not in self.domain_dic.keys()) and (d[1] in self.domain_dic.keys()):
                self.domain_dic[d[0]] = self.domain_dic[d[1]]
        f.close

        # original data that have domain feature
        f = open('./data/clean_range.txt')
        for line in f.readlines():
            line = line.strip('\n')
            d = line.split(' ')
            d[0] = d[0].strip('\r')
            d[1] = d[1].strip('\r')
            self.range_dic[d[0]] = d[1]
            if d[1] not in self.rt:
                self.rt.append(d[1])
        f.close

        # subPropertyOf data to generate range feature from parent property
        f = open('./data/clean_subPropertyOf.txt')
        for line in f.readlines():
            line = line.strip('\n')
            d = line.split(' ')
            d[0] = d[0].strip('\r')
            d[1] = d[1].strip('\r')
            if (d[0] not in self.range_dic.keys()) and (d[1] in self.range_dic.keys()):
                self.range_dic[d[0]] = self.range_dic[d[1]]
        f.close

        word2vec_model_path = './download/Glove/glove.6B.50d.txt'
        word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False,
                                                           unicode_errors='ignore')
        self.property_vec = word2vec_model.get_vector('property')
        self.class_vec = word2vec_model.get_vector('class')

    def generateDomainTrainData(self):
        """
        Create domain data for training
        :return: 
        """
        X = []
        Y = []
        for _key, _value in self.domain_dic.iteritems():
            words_in_key = []
            ws = _key.split('/')
            ww = ws[-2:]
            for ii in ww:
                ti = ii.split('#')
                for it in ti:
                    words_in_key.append(it)
            # words_in_key = _key.split('/')
            # words_in_key = words_in_key[-1]
            # words_in_key = words_in_key.split('#')
            _value = _value.strip('\r')
            clean_words = []
            for item in words_in_key:
                if item == "http:" or item == 'dbpedia.org' or item == '' or item == 'www.w3.org':
                    continue
                else:
                    pattern = "[A-Z]"
                    new_string = re.sub(pattern, lambda x: " " + x.group(0), item)
                    ss = new_string.split(' ')
                    for _s in ss:
                        if _s != ' ' and _s != '':
                            _s = _s.strip('-')
                            _s = _s.strip('_')
                            clean_words.append(_s)
            _ = ' '.join(clean_words)
            print _
            res_vec = self.trans.transform2onevec(_)
            X.append(res_vec)
            pos = self.domain_types.index(_value)
            onehot = np.zeros(269, dtype='float32')
            onehot[pos] = 1
            onehot = np.array(onehot)
            onehot = onehot.astype('float32')
            Y.append(onehot)
        np.save('./data/domain_train_x', X)
        np.save('./data/domain_train_y', Y)

    def generateRangeTrainData(self):
        """
        Create range data for training
        :return: 
        """
        X = []
        Y = []
        for _key, _value in self.range_dic.iteritems():
            words_in_key = []
            ws = _key.split('/')
            ww = ws[-2:]
            for ii in ww:
                ti = ii.split('#')
                for it in ti:
                    words_in_key.append(it)
            # words_in_key = _key.split('/')
            # words_in_key = words_in_key[-1]
            # words_in_key = words_in_key.split('#')
            _value = _value.strip('\r')
            clean_words = []
            for item in words_in_key:
                if item == "http:" or item == 'dbpedia.org' or item == '' or item == 'www.w3.org':
                    continue
                else:
                    pattern = "[A-Z]"
                    new_string = re.sub(pattern, lambda x: " " + x.group(0), item)
                    ss = new_string.split(' ')
                    for _s in ss:
                        if _s != ' ' and _s != '':
                            _s = _s.strip('-')
                            _s = _s.strip('_')
                            clean_words.append(_s)
            _ = ' '.join(clean_words)
            print _
            res_vec = self.trans.transform2onevec(_)
            X.append(res_vec)
            pos = self.range_types.index(_value)
            onehot = np.zeros(206, dtype='float32')
            onehot[pos] = 1
            onehot = np.array(onehot)
            onehot = onehot.astype('float32')
            Y.append(onehot)
        np.save('./data/range_train_x', X)
        np.save('./data/range_train_y', Y)
#
# if __name__ == '__main__':
#     r = Features()
#     r.generateDomainTrainData()
#     r.generateRangeTrainData()
#     # print r.getUriVec('http://dbpedia.org/property/combatant')
#     # print r.getDomainVec('http://dbpedia.org/property/combatant')
#     # print r.getTypeVec('http://dbpedia.org/property/combatant')
#     # print r.getUriVec('http://dbpedia.org/ontology/birthPlace')
#     # print r.getDomainVec('http://dbpedia.org/ontology/birthPlace')
#     # print r.getRangeVec('http://dbpedia.org/ontology/birthPlace')
#     # print r.getTypeVec('http://dbpedia.org/ontology/birthPlace')
#     #
#     # print r.getUriVec('http://dbpedia.org/property/birthPlace')
#     # print r.getDomainVec('http://dbpedia.org/property/birthPlace')
#     # print r.getRangeVec('http://dbpedia.org/property/birthPlace')
#     # print r.getTypeVec('http://dbpedia.org/property/birthPlace')