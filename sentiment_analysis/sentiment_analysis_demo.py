import gradio as gr

img_path = "./sentiment_analysis/images"
overall_senti_barchart = "overall_senti_barchart.png"
senti_VICustomerCare = "senti_VICustomerCare.png"
senti_event_IPLAUCTION = "senti_event_IPLAUCTION.png"
senti_influential_users_piechart = "senti_influential_users_piechart.png"
senti_ipl2022 = "senti_ipl2022.png"
senti_lsg = "senti_lsg.png"
senti_match_CSKvsRSB = "senti_match_CSKvsRSB.png"
senti_match_GTvsDC = "senti_match_GTvsDC.png"
senti_match_KKRvsPBKS = "senti_match_KKRvsPBKS.png"
senti_match_LSGvsCSK = "senti_match_LSGvsCSK.png"
senti_match_MIvsCSK = "senti_match_MIvsCSK.png"
senti_match_MIvsPBKS = "senti_match_MIvsPBKS.png"
senti_match_RRvsGT = "senti_match_RRvsGT.png"
senti_match_RCBvKKR = "senti_match_RCBvKKR.png"
senti_player_MSDoni = "senti_player_MSDoni.png"
senti_player_doni = "senti_player_doni.png"
senti_player_virat_kohli = "senti_player_virat kohli.png"
senti_player_viratkohli2 = "senti_player_viratkohli2.png"
senti_team_Delhi_Capitals = "senti_team_Delhi Capitals (DC).png"
senti_team_GT = "senti_team_GT.png"
senti_team_PBKS = "senti_team_PBKS.png"
senti_team_gujarat = "senti_team_gujarat.png"
senti_team_punjab_teams = "senti_team_punjab teams.png"

def load_sentiment_analysis_demo():
    with gr.Blocks() as sent_analy:
        gr.Markdown("# Sentiment Analysis")
        gr.Markdown('''
            ## Description
            Our sentiment analysis graph presents insights derived from analysing IPL-related tweets posted during the 2022 season. Using the VADER sentiment analysis tool, we categorised tweets into positive, neutral, and negative sentiments. This provides a comprehensive overview of fan perceptions and reactions surrounding IPL matches, teams, and players.
            ## Significance
            Understanding the sentiments expressed in IPL tweets is invaluable for stakeholders, including IPL franchises, marketers, and fan engagement teams. These insights inform strategic decision-making, enhance marketing campaigns, and guide efforts to foster a positive fan experience.
            ''')
        
        gr.Markdown("# General Sentiment")
        with gr.Row():
            with gr.Column():
                sa_1_img = gr.Image(value=f"{img_path}/{senti_ipl2022}")
                sa_1_md = gr.Markdown("The overall compound sentiment score of IPL tweets in 2022 indicates a balanced mix of positive, negative, and neutral sentiments. Tweets with scores around 0 express neither strongly positive nor negative attitudes. They may represent factual statements or expressions of indifference.")
            with gr.Column():
                sa_2_img = gr.Image(value=f"{img_path}/{senti_event_IPLAUCTION}")
                sa_2_md = gr.Markdown("The IPL auction is a crucial event where franchises bid to buy players, shaping IPL teams and the cricketing spectacle.")
        with gr.Row():
            with gr.Column():
                sa_3_img = gr.Image(value=f"{img_path}/{senti_influential_users_piechart}")
                sa_3_md = gr.Markdown("Pie Chart of Sentiment Analysis for Influential Users with >100k Followers: This chart provides insights into attitudes and opinions expressed by key influencers within the IPL community. Influential users' sentiments can shape perceptions of IPL aspects like teams, players, matches, and league management, affecting the league's image and reputation.")
            with gr.Column():
                sa_4_img = gr.Image(value=f"{img_path}/{overall_senti_barchart}")
                sa_4_md = gr.Markdown("Sentiment Distribution in Tweets Regarding IPL2022 ")

        gr.Markdown("# Match Sentiment")
        with gr.Row():
            with gr.Column():
                sa_5_title = gr.Markdown("## Chennai Super Kings (CSK) vs Royal Challengers Bengaluru (RCB)")
                sa_5_img = gr.Image(value=f"{img_path}/{senti_match_CSKvsRSB}")
            with gr.Column():
                sa_6_title = gr.Markdown("## Gujarat Titans (GT) vs Delhi Capitals (DC)")
                sa_6_img = gr.Image(value=f"{img_path}/{senti_match_GTvsDC}")
        with gr.Row():
            with gr.Column():
                sa_7_title = gr.Markdown("## Kolkata Knight Riders (KKR) vs Punjab Kings (PBKS)")
                sa_7_img = gr.Image(value=f"{img_path}/{senti_match_KKRvsPBKS}")
            with gr.Column():
                sa_8_title = gr.Markdown("## Lucknow Super Giants (LSG) vs Chennai Super Kings (CSK)")
                sa_8_img = gr.Image(value=f"{img_path}/{senti_match_LSGvsCSK}")
        with gr.Row():
            with gr.Column():
                sa_9_title = gr.Markdown("## Mumbai Indians (MI) vs Chennai Super Kings (CSK)")
                sa_9_img = gr.Image(value=f"{img_path}/{senti_match_MIvsCSK}")
            with gr.Column():
                sa_10_title = gr.Markdown("## Mumbai Indians (MI) vs Punjab Kings (PBKS)")
                sa_10_img = gr.Image(value=f"{img_path}/{senti_match_MIvsPBKS}")
        with gr.Row():
            with gr.Column():
                sa_11_title = gr.Markdown("## Rajasthan Royals (RR) vs Gujarat Titans (GT)")
                sa_11_img = gr.Image(value=f"{img_path}/{senti_match_RRvsGT}")
            with gr.Column():
                sa_12_title = gr.Markdown("## Royal Challengers Bengaluru (RCB) vs Kolkata Knight Riders (KKR)")
                sa_12_img = gr.Image(value=f"{img_path}/{senti_match_RCBvKKR}")

        gr.Markdown("# Player Sentiment")
        with gr.Row():
            with gr.Column():
                sa_13_title = gr.Markdown("## Mahendra Singh Dhoni")
                sa_13_img = gr.Image(value=f"{img_path}/{senti_player_MSDoni}")
                sa_13_md = gr.Markdown("MS Dhoni is known for his unorthodox captaincy of the Indian cricket team, approachability and has earned a reputation of being a successful leader.")
            with gr.Column():
                sa_14_title = gr.Markdown("## Virat Kohli")
                sa_14_img = gr.Image(value=f"{img_path}/{senti_player_viratkohli2}")
                sa_14_md = gr.Markdown("Virat Kohli who took over as RCB's full-time captain in 2013, led them in 140 matches, winning 66 and losing as many as 70.")

        gr.Markdown("# Team Sentiment")
        with gr.Row():
            with gr.Column():
                sa_15_title = gr.Markdown("## Delhi Capitals (DC)")
                sa_15_img = gr.Image(value=f"{img_path}/{senti_team_Delhi_Capitals}")
            with gr.Column():
                sa_16_title = gr.Markdown("## Gujarat Titans (GT)")
                sa_16_img = gr.Image(value=f"{img_path}/{senti_team_GT}")
        with gr.Row():
            with gr.Column():
                sa_17_title = gr.Markdown("## Punjab Kings (PBKS)")
                sa_17_img = gr.Image(value=f"{img_path}/{senti_team_PBKS}")
            with gr.Column():
                sa_18_title = gr.Markdown("## Lucknow Super Giants (LSG) ")
                sa_18_img = gr.Image(value=f"{img_path}/{senti_lsg}")
                
    return sent_analy