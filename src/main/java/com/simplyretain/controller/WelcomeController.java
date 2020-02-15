package com.simplyretain.controller;

import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.client.RestTemplate;

import com.simplyretain.domain.Characteristics;
import com.simplyretain.domain.User;
import com.simplyretain.domain.UsersFactory;

@Controller
public class WelcomeController {
	
	private static final String URI = "http://127.0.0.1";
	private static final String API_URI = URI + ":5000/";
	
	@RequestMapping("/")
	public String index(Model model) {
		 model.addAttribute("user", new User());
		return "index";
	}
	
	@RequestMapping("/validate")
	public String validateUser(@ModelAttribute("user") User user, Model model) {
		if(UsersFactory.isAuthenticated(user)) {
			return bubbles(model);
		}
		return "index";
	}
	
	@RequestMapping("/bubbles")
	public String bubbles(Model model) {
		model.addAttribute("character", new Characteristics());
		return "bubbles";
	}
	
	@RequestMapping("/postchar")
	public String receiveCharacterictics(@ModelAttribute("character") Characteristics character, Model model) {
		// send this and get the details
		model.addAttribute("char", character);
		return results(character,model);
	}
	
	@RequestMapping("/result")
	public String results(@ModelAttribute("character") Characteristics character,Model model) {
		String[] images = getImage(character);
		String payload = getRetention();
		String decisionTree = getTree(payload);
		model.addAttribute("data", payload);
		model.addAttribute("tree_imag", "data:image/png;base64, "+decisionTree.replaceAll("\"", ""));
		model.addAttribute("image_1", "data:image/png;base64, "+images[0].replaceAll("\"", ""));
		model.addAttribute("image_2", "data:image/png;base64, "+images[1].replaceAll("\"", ""));
		return "result";
	}
	
	@RequestMapping("/retention")
	public String getRetentiondata(Model model) {
		return getRetention();
	}
	
	private String[] getImage(Characteristics character) {
	    final String uri = API_URI + "get_image";
	    RestTemplate restTemplate = new RestTemplate();
		String result = restTemplate.postForObject(uri, character,String.class);
	    return result.split("@@@");
	}
	
	private String getTree(String payload) {
		HttpHeaders headers = new HttpHeaders();
		headers.setContentType(MediaType.APPLICATION_JSON);

		HttpEntity<String> entity = new HttpEntity<String>(payload,headers);
	    final String uri = API_URI + "get_tree";
	    RestTemplate restTemplate = new RestTemplate();
	    String result = restTemplate.postForObject(uri, entity,String.class);
	    return result;
	}
	
	private String getRetention() {
	    final String uri = API_URI + "retention/millenial/14";
	    RestTemplate restTemplate = new RestTemplate();
	    String result = restTemplate.getForObject(uri, String.class);
	    return result;
	}
}