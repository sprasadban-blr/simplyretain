package com.simplyretain.domain;

import java.util.HashMap;
import java.util.Map;

public final class UsersFactory {
	
	private static Map<String, String> users;
	
	static {
		users = new HashMap<>();
		users.put("admin", "admin");
	}
	
	public static boolean isAuthenticated(User user) {
		if(users.containsKey(user.getUsername())) {
			return users.get(user.getUsername()).equals(user.getPassword());
		}
		return false;
	}

}
