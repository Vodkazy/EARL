#include<bits/stdc++.h>
using namespace std;

int main(){
	/***********************
	*
	* Extract the feature of rdf 
	*
	************************
	freopen("word2vec-api-master/dbpedia_ontology.nt","r",stdin);
	freopen("type.txt","w",stdout);
	string s;
	while(getline(cin,s)){
		if(s.find("#type")!=-1)
			cout<<s<<endl;
	}
	************************/ 
	
	
	/***********************
	*
	* Extract uri:domain pair
	*
	************************
	freopen("range.txt","r",stdin);
	freopen("clean_range.txt","w",stdout);
	string s;
	while(getline(cin,s)){
		int pos = 0;
		pos = s.find('>');
		for(int i=1;i<pos;i++){
			if(s[i]=='>')break;
			cout<<s[i];	
		}
		cout<<" ";
		
		pos = 0;
		pos = s.rfind('>');
		for(int i=pos-1;i>=0;i--){
			if(s[i]=='/')break;
			pos = i;
		}
		if(s.find('#',pos=pos)!=-1){
			pos = s.find('#',pos=pos)+1;
		}
		for(int i=pos;i<s.length();i++){
			if(s[i]=='>')break;
			cout<<s[i];	
		}
		cout<<endl;
	}	
	************************/ 


	/***********************
	*
	* Extract uri:parent_uri pair
	*
	************************
	freopen("subPropertyOf.txt","r",stdin);
	freopen("clean_subPropertyOf.txt","w",stdout);
	string s;
	while(getline(cin,s)){
		int pos = 0;
		pos = s.find('>');
		for(int i=1;i<pos;i++){
			if(s[i]=='>')break;
			cout<<s[i];	
		}
		cout<<" ";
		
		pos = 0;
		pos = s.rfind('>');
		for(int i=pos-1;i>=0;i--){
			if(s[i]=='<')break;
			pos = i;
		}
		for(int i=pos;i<s.length();i++){
			if(s[i]=='>')break;
			cout<<s[i];	
		}
		cout<<endl;
	}	
	************************/ 


	/***********************
	*
	* Extract uri:type pair 
	*
	***********************/
	freopen("type.txt","r",stdin);
	freopen("clean_type.txt","w",stdout);
	string s;
	while(getline(cin,s)){
		int pos = 0;
		pos = s.find('>');
		for(int i=0;i<pos;i++){
			if(s[i]=='>')break;
			cout<<s[i];	
		}
		cout<<" ";
		
		pos = 0;
		pos = s.rfind('>');
		for(int i=pos-1;i>=0;i--){
			if(s[i]=='<')break;
			pos = i;
		}
		if(s.find('#',pos=pos)!=-1){
			pos = s.find('#',pos=pos)+1;
		}
		for(int i=pos;i<s.length();i++){
			if(s[i]=='>')break;
			cout<<s[i];	
		}
		cout<<endl;
	}	
	return 0;
}
