package com.Selenium;

import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

public class demoTemplate {

		public static void main(String[] args) {
				String driverPath = "webdriver.chrome.driver";
				String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe";

				System.setProperty(driverPath, driverURL);
				WebDriver driver = new ChromeDriver(); 
				String baseUrl = "";
				try
				{	
					System.out.println("************* Try Block ***********************");
					 driver.get(baseUrl); 
					 
					 //Wrtie Code here
				}
				catch(Exception e)
				{
					System.out.println("************* Catch Block ***********************");
					System.out.println("Exception Found in Scripts " + e);
				}
				finally
				{
				   System.out.println("************* Finally Block ***********************");	
				   driver.close();
				   driver.quit();
				 }
				

			}

		}

	