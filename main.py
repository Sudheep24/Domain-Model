from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample data (replace with actual file read operations if needed)
domainskills = {
    "1": "HTML",
    "2": "CSS",
    "3": "JavaScript",
    "4": "React.js",
    "5": "Angular.js",
    "6": "Node.js",
    "7": "Express.js",
    "8": "Python",
    "9": "Ruby on Rails",
    "10": "Java",
    "11": "SQL",
    "12": "NoSQL",
    "13": "MongoDB",
    "14": "Git",
    "15": "GitHub",
    "16": "Docker",
    "17": "Kubernetes",
    "18": "CI/CD pipelines",
    "19": "R",
    "20": "Pandas",
    "21": "NumPy",
    "22": "Scikit-Learn",
    "23": "TensorFlow",
    "24": "Keras",
    "25": "Matplotlib",
    "26": "Seaborn",
    "27": "Tableau",
    "28": "Hadoop",
    "29": "Spark",
    "30": "PyTorch",
    "31": "Flask",
    "32": "Regression",
    "33": "Classification",
    "34": "Clustering",
    "35": "Firewalls",
    "36": "VPNs",
    "37": "IDS/IPS",
    "38": "Encryption algorithms",
    "39": "PKI",
    "40": "Penetration testing",
    "41": "Risk analysis",
    "42": "Forensics",
    "43": "Log analysis",
    "44": "Wireshark",
    "45": "Metasploit",
    "46": "Nessus",
    "47": "C++",
    "48": "Agile",
    "49": "Scrum",
    "50": "MVC",
    "51": "Singleton",
    "52": "Factory",
    "53": "Unit Testing",
    "54": "Integration Testing",
    "55": "Selenium",
    "56": "AWS",
    "57": "Azure",
    "58": "Google Cloud",
    "59": "EC2",
    "60": "S3",
    "61": "Lambda",
    "62": "Terraform",
    "63": "CloudFormation",
    "64": "VPC",
    "65": "Load Balancers",
    "66": "DNS",
    "67": "IAM",
    "68": "Security Groups",
    "69": "Encryption",
    "70": "Vue.js",
    "71": "Django",
    "72": "GraphQL",
    "73": "Jenkins",
    "74": "Ansible",
    "75": "Puppet",
    "76": "Nagios",
    "77": "Prometheus",
    "78": "Grafana",
    "79": "Bitbucket",
    "80": "Unity",
    "81": "Unreal Engine",
    "82": "OpenGL",
    "83": "DirectX",
    "84": "Havok",
    "85": "Bullet",
    "86": "Multiplayer game networking",
    "87": "APIs",
    "88": "Windows",
    "89": "Linux",
    "90": "MacOS",
    "91": "TCP/IP",
    "92": "PC assembly",
    "93": "Peripheral troubleshooting",
    "94": "OS issues",
    "95": "Application errors",
    "96": "TeamViewer",
    "97": "Remote Desktop"
}
domain_skills = {
  "Full Stack Software Development": [1, 2, 3, 4, 6, 7, 11, 14, 16],
  "Data Science": [8, 19, 20, 21, 22, 23, 24, 25, 26],
  "Machine Learning Engineering": [8, 20, 21, 22, 23, 24, 30, 32, 33, 34],
  "Cybersecurity": [35, 36, 37, 38, 39, 40, 41, 42, 43],
  "Software Engineering": [1, 3, 10, 47, 48, 49, 50, 51, 53],
  "Cloud Engineering": [16, 17, 18, 56, 57, 58, 62, 63, 65],
  "Web Development": [1, 2, 3, 4, 5, 14, 70, 71, 72],
  "DevOps": [16, 18, 56, 57, 58, 73, 74, 75, 76],
  "Game Development": [47, 80, 81, 82, 83, 84, 85, 86],
  "IT Support": [88, 89, 90, 91, 92, 93, 94, 95, 96]
}
domain_difficulty_marks = {
    "Full Stack Software Development": (80, 100),
    "Data Science": (70, 90),
    "Machine Learning Engineering": (85, 100),
    "Cybersecurity": (90, 100),
    "Software Engineering": (75, 95),
    "Cloud Engineering": (70, 90),
    "Web Development": (60, 80),
    "DevOps": (80, 95),
    "Game Development": (65, 85),
    "IT Support": (50, 70)
}
def map_skills_to_domain(skill_ids):
    domain_skill_counts = {domain: 0 for domain in domain_skills}
    
    for domain, skills in domain_skills.items():
        skill_set = set(skills)
        user_skill_set = set(skill_ids)
        common_skills = skill_set.intersection(user_skill_set)
        domain_skill_counts[domain] = len(common_skills)
    
    max_skill_count = max(domain_skill_counts.values())
    top_domains = [domain for domain, count in domain_skill_counts.items() if count == max_skill_count]
    
    if len(top_domains) == 1:
        return top_domains[0]
    
    def get_domain_rank(domain, math_mark, aptitude_mark, science_mark):
        min_mark, max_mark = domain_difficulty_marks[domain]
        rank = 0
        if math_mark >= min_mark:
            rank += 1
        if aptitude_mark >= min_mark:
            rank += 1
        if science_mark >= min_mark:
            rank += 1
        return rank
    
    user_marks = {
        "Math": 85,  
        "Aptitude": 80,
        "Science": 90
    }
    
    ranked_domains = sorted(
        top_domains,
        key=lambda domain: get_domain_rank(domain, user_marks["Math"], user_marks["Aptitude"], user_marks["Science"]),
        reverse=True
    )
    
    return ranked_domains[0]

@app.route('/recommend-domain', methods=['POST'])
def recommend_domain():
    data = request.json
    skill_ids = data.get('skill_ids', [])
    math_mark = data.get('math_mark', 0)
    aptitude_mark = data.get('aptitude_mark', 0)
    science_mark = data.get('science_mark', 0)
    
    recommended_domain = map_skills_to_domain(skill_ids)
    return jsonify({'recommended_domain': recommended_domain})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)