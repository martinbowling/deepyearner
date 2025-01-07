#!/usr/bin/env python3
"""
CLI tool for managing user analysis.
"""
import click
import json
import csv
from typing import List
from twitter_utils import get_twitter_client
from user_analysis import UserAnalysisSystem, AnalysisSource
from user_analyzer import UserAnalyzer

@click.group()
def cli():
    """User analysis management tool"""
    pass

@cli.command()
@click.argument('username')
@click.option('--priority', default=1, help='Analysis priority (1-5)')
def add_user(username: str, priority: int):
    """Add a single user to analyze"""
    twitter = get_twitter_client()
    analysis_system = UserAnalysisSystem()
    
    # Get user info
    user = twitter.get_user_by_username(username)
    if not user or 'data' not in user:
        click.echo(f"Error: Could not find user {username}")
        return
    
    user_id = user['data']['id']
    
    # Add to analysis queue
    if analysis_system.add_target_user(
        user_id=user_id,
        username=username,
        source=AnalysisSource.MANUAL,
        priority=priority
    ):
        click.echo(f"Added {username} to analysis queue")
    else:
        click.echo(f"Note: {username} was recently analyzed or already in queue")

@cli.command()
@click.argument('list_id')
@click.option('--process-members/--no-process-members', default=True, 
              help='Immediately process list members')
def add_list(list_id: str, process_members: bool):
    """Add a Twitter list to monitor"""
    twitter = get_twitter_client()
    analysis_system = UserAnalysisSystem()
    
    try:
        # Get list info
        list_info = twitter.get_list(list_id)
        if not list_info or 'data' not in list_info:
            click.echo(f"Error: Could not find list {list_id}")
            return
        
        list_data = list_info['data']
        list_name = list_data.get('name', 'Unknown List')
        owner_id = list_data.get('owner_id', 'unknown')
        
        # Add list to tracking
        if analysis_system.add_target_list(list_id, list_name, owner_id):
            click.echo(f"Added list {list_name} ({list_id}) to monitoring")
            
            if process_members:
                # Get list members
                members = twitter.get_list_members(list_id)
                if members and 'data' in members:
                    click.echo(f"Processing {len(members['data'])} list members...")
                    
                    for member in members['data']:
                        if analysis_system.add_target_user(
                            user_id=member['id'],
                            username=member['username'],
                            source=AnalysisSource.LIST,
                            source_list_id=list_id
                        ):
                            click.echo(f"Added {member['username']} from list")
                else:
                    click.echo("No members found in list")
        else:
            click.echo(f"Error: Could not add list {list_id}")
            
    except Exception as e:
        click.echo(f"Error processing list: {str(e)}")

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['csv', 'json']), default='csv',
              help='Input file format')
@click.option('--priority', default=1, help='Analysis priority (1-5)')
def batch_import(file_path: str, format: str, priority: int):
    """Import users from a file (CSV or JSON)"""
    twitter = get_twitter_client()
    analysis_system = UserAnalysisSystem()
    
    try:
        users = []
        if format == 'csv':
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    username = row.get('username') or row.get('user') or row.get('handle')
                    if username:
                        users.append(username.strip('@'))
        else:  # JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            users.append(item.strip('@'))
                        elif isinstance(item, dict):
                            username = item.get('username') or item.get('user') or item.get('handle')
                            if username:
                                users.append(username.strip('@'))
        
        click.echo(f"Found {len(users)} users to import...")
        
        for username in users:
            # Get user info
            user = twitter.get_user_by_username(username)
            if user and 'data' in user:
                user_id = user['data']['id']
                if analysis_system.add_target_user(
                    user_id=user_id,
                    username=username,
                    source=AnalysisSource.MANUAL,
                    priority=priority
                ):
                    click.echo(f"Added {username} to analysis queue")
            else:
                click.echo(f"Could not find user {username}")
                
    except Exception as e:
        click.echo(f"Error importing users: {str(e)}")

@cli.command()
@click.option('--limit', default=10, help='Number of users to analyze')
def analyze_pending(limit: int):
    """Analyze pending users"""
    twitter = get_twitter_client()
    analysis_system = UserAnalysisSystem()
    analyzer = UserAnalyzer(twitter)
    
    # Get pending users
    pending = analysis_system.get_pending_users(limit)
    if not pending:
        click.echo("No pending users to analyze")
        return
    
    click.echo(f"Analyzing {len(pending)} users...")
    
    for user in pending:
        click.echo(f"\nAnalyzing {user['username']}...")
        
        # Perform analysis
        analysis = analyzer.analyze_user(
            user_id=user['user_id'],
            username=user['username'],
            source=AnalysisSource(user['source'])
        )
        
        if analysis:
            # Save results
            analysis_system.save_analysis(analysis)
            
            # Display results
            click.echo(f"Results for {user['username']}:")
            click.echo(f"Overall Score: {analysis.overall_score:.2f}")
            click.echo(f"Recommendation: {analysis.recommendation}")
            click.echo(f"Notes: {analysis.notes}")
        else:
            click.echo(f"Error analyzing {user['username']}")

@cli.command()
@click.argument('username')
def show_analysis(username: str):
    """Show analysis results for a user"""
    twitter = get_twitter_client()
    analysis_system = UserAnalysisSystem()
    
    # Get user info
    user = twitter.get_user_by_username(username)
    if not user or 'data' not in user:
        click.echo(f"Error: Could not find user {username}")
        return
    
    user_id = user['data']['id']
    
    # Get analysis history
    history = analysis_system.get_analysis_history(user_id)
    if history:
        click.echo(f"\nAnalysis results for {username}:")
        click.echo(f"Last analyzed: {history['last_analyzed']}")
        click.echo(f"Engagement rate: {history['engagement_rate']:.2f}")
        click.echo(f"Topic alignment: {history['topic_alignment']:.2f}")
        click.echo(f"Interaction quality: {history['interaction_quality']:.2f}")
        click.echo(f"Overall score: {history['overall_score']:.2f}")
        click.echo(f"Recommendation: {history['recommendation']}")
        click.echo(f"Notes: {history['notes']}")
    else:
        click.echo(f"No analysis history found for {username}")

if __name__ == '__main__':
    cli() 